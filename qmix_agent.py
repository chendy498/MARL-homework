import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 个体 Q 网络（Actor）
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # 输出 Q 值


# QMIX 混合网络
class Mixer(nn.Module):
    def __init__(self, n_agents, state_dim):
        super(Mixer, self).__init__()
        self.n_agents = n_agents
        self.hyper_w1 = nn.Linear(state_dim, n_agents * 32)
        self.hyper_w2 = nn.Linear(state_dim, 32)
        self.hyper_b1 = nn.Linear(state_dim, 32)
        self.hyper_b2 = nn.Linear(state_dim, 1)

    def forward(self, q_values, states):
        batch_size = q_values.shape[0]
        states = states.view(batch_size, -1)

        # 计算权重
        w1 = torch.abs(self.hyper_w1(states)).view(batch_size, self.n_agents, 32)
        w2 = torch.abs(self.hyper_w2(states)).view(batch_size, 32, 1)
        b1 = self.hyper_b1(states).view(batch_size, 1, 32)
        b2 = self.hyper_b2(states).view(batch_size, 1, 1)

        # 计算全局 Q 值
        q_values = q_values.view(batch_size, 1, self.n_agents)
        hidden = F.elu(torch.bmm(q_values, w1) + b1)
        q_total = torch.bmm(hidden, w2) + b2
        return q_total.squeeze(-1)


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, state, actions, rewards, next_obs, next_state, dones):
        self.buffer.append((obs, state, actions, rewards, next_obs, next_state, dones))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, states, actions, rewards, next_obs, next_states, dones = zip(*batch)
        return (np.array(obs), np.array(states), np.array(actions),
                np.array(rewards), np.array(next_obs), np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


# QMIX 智能体
class QMIXAgent:
    def __init__(self, obs_dim, act_dim, state_dim, n_agents, lr=3e-4, gamma=0.99, tau=0.01, capacity=10000):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau

        # 个体 Q 网络
        self.q_networks = [QNetwork(obs_dim, act_dim).to(device) for _ in range(n_agents)]
        self.target_q_networks = [QNetwork(obs_dim, act_dim).to(device) for _ in range(n_agents)]

        # 混合网络（Mixer）
        self.mixer = Mixer(n_agents, state_dim).to(device)
        self.target_mixer = Mixer(n_agents, state_dim).to(device)

        # 复制参数到目标网络
        for target_q, q in zip(self.target_q_networks, self.q_networks):
            target_q.load_state_dict(q.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        # 优化器
        self.q_optimizers = [optim.Adam(q.parameters(), lr=lr) for q in self.q_networks]
        self.mixer_optimizer = optim.Adam(self.mixer.parameters(), lr=lr)

        # 经验回放缓冲区
        self.memory = ReplayBuffer(capacity)

    def act(self, obs, explore=True):
        obs = torch.FloatTensor(obs).to(device)
        actions = []
        for i, q_network in enumerate(self.q_networks):
            q_values = q_network(obs[i].unsqueeze(0))
            if explore:
                action = np.random.choice(self.act_dim)  # 采用 ε-greedy 探索（这里简化为随机采样）
            else:
                action = torch.argmax(q_values, dim=-1).item()
            actions.append(action)
        return actions

    def store(self, obs, state, actions, rewards, next_obs, next_state, dones):
        self.memory.push(obs, state, actions, rewards, next_obs, next_state, dones)

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # 采样经验
        obs, states, actions, rewards, next_obs, next_states, dones = self.memory.sample(batch_size)
        obs = torch.FloatTensor(obs).to(device)
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device).unsqueeze(-1)
        next_obs = torch.FloatTensor(next_obs).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device).unsqueeze(-1)

        # 计算个体 Q 值
        q_values = torch.stack([q(obs[:, i, :]).gather(1, actions[:, i].unsqueeze(-1)) for i, q in enumerate(self.q_networks)], dim=1)

        # 计算目标 Q 值
        with torch.no_grad():
            target_q_values = torch.stack([target_q(next_obs[:, i, :]).max(dim=-1, keepdim=True)[0] for i, target_q in enumerate(self.target_q_networks)], dim=1)
            q_total_target = self.target_mixer(target_q_values.squeeze(-1), next_states)
            target = rewards.sum(dim=1, keepdim=True) + self.gamma * q_total_target * (1 - dones)

        # 计算 Mixer 输出的全局 Q 值
        q_total = self.mixer(q_values.squeeze(-1), states)

        # 计算损失并优化
        loss = F.mse_loss(q_total, target.detach())
        self.mixer_optimizer.zero_grad()
        loss.backward()
        self.mixer_optimizer.step()

    def save(self, filepath):
        torch.save({'mixer_state_dict': self.mixer.state_dict()}, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=device)
        self.mixer.load_state_dict(checkpoint['mixer_state_dict'])
        print(f"Model loaded from {filepath}")


# 测试 QMIX
if __name__ == "__main__":
    agent = QMIXAgent(obs_dim=10, act_dim=5, state_dim=20, n_agents=3)
    obs = np.random.randn(3, 10)
    action = agent.act(obs)
    print(f"示例动作: {action}")
