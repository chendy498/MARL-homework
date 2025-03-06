import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim * n_agents + action_dim * n_agents, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# MADDPG智能体
class MADDPGAgent:
    def __init__(self, obs_dim, act_dim, state_dim, n_agents, lr=0.001, gamma=0.99, tau=0.01):
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(state_dim, act_dim, n_agents).to(device)
        self.actor_target = Actor(obs_dim, act_dim).to(device)
        """self.actor_target 是一个目标网络（Target Network），在强化学习中用于稳定训练过程。
        具体来说，它是 self.actor 的一个副本，但其参数更新速度较慢，用于提供稳定的目标值
        这种目标网络的设计是深度强化学习（如 DDPG 和 MADDPG）中的标准技术，用于解决训练过程中的不稳定性和收敛问题
        之前用PPO算法出问题的解决方案"""

        self.critic_target = Critic(state_dim, act_dim, n_agents).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict()) #加载模型参数
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau#软更新
        self.memory = deque(maxlen=10000)#双向队列，用于存储经验
        self.act_dim = act_dim

        self.agents=[]

    def act(self, obs):
        obs = torch.FloatTensor(obs).to(device)
        with torch.no_grad():#不进行梯度计算
            probs = self.actor(obs)#返回一个动作的概率分布
        return torch.multinomial(probs, 1).item()#采样一个动作

    def train(self, batch_size, all_obs, all_actions, all_next_obs):
        if len(self.memory) < batch_size:#如果经验池中的经验数量小于batch_size，不进行训练
            return
        batch = random.sample(self.memory, batch_size)#一次取batch_size大小的批
        states, actions, rewards, next_states, dones = zip(*batch)#解压
        #写入GPU
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        all_obs = torch.FloatTensor(all_obs).to(device)
        all_actions = torch.FloatTensor(all_actions).to(device)
        all_next_obs = torch.FloatTensor(all_next_obs).to(device)

        # Critic 更新
        # 计算目标Q值
        # next_actions = [agent.actor_target(next_states[:, i]) for i, agent in enumerate(self.agents)]#预测下一状态（next_states）对应的动作
        next_actions = self.actor_target(next_states)
        # next_actions = torch.stack(next_actions, dim=1)#将多个张量拼接在一起，dim=1表示在第二维度上拼接，拼接成一列矩阵
        q_targets = rewards + self.gamma * self.critic_target(all_next_obs, next_actions) * (1 - dones)
        q_values = self.critic(all_obs, all_actions)
        critic_loss = nn.MSELoss()(q_values, q_targets.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor 更新
        pred_actions = self.actor(states)
        actor_loss = -self.critic(all_obs, pred_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_target(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))