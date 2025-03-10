import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random

# 设置设备：如果有 GPU 则使用 cuda，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 工具函数：用于 Gumbel-Softmax 采样 ---
def sample_gumbel(shape, eps=1e-20):
    """从 Gumbel(0,1) 分布中采样，用于离散动作选择"""
    U = torch.rand(shape).to(device)  # 生成均匀分布随机数
    return -torch.log(-torch.log(U + eps) + eps)  # 转换为 Gumbel 分布

def gumbel_softmax_sample(logits, temperature=1.0):
    """从 Gumbel-Softmax 分布中采样，生成概率分布"""
    y = logits + sample_gumbel(logits.shape)  # 添加 Gumbel 噪声
    return F.softmax(y / temperature, dim=-1)  # 通过温度参数控制分布的平滑度

def gumbel_softmax(logits, temperature=1.0, hard=False):
    """从 Gumbel-Softmax 分布采样，可选择硬采样（one-hot）或软采样（概率分布）"""
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        # 硬采样：将最大值位置设为 1，其余为 0
        y_hard = torch.zeros_like(y).scatter_(-1, torch.argmax(y, dim=-1, keepdim=True), 1.0)
        return (y_hard - y).detach() + y  # 保持梯度可传递
    return y

# --- Actor 网络：策略网络，用于生成动作 ---
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        """初始化 Actor 网络
        参数：
            input_dim: 输入维度（观测空间大小）
            output_dim: 输出维度（动作空间大小）
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 第一层全连接，输入到 128 维
        self.fc2 = nn.Linear(128, 64)         # 第二层全连接，128 到 64 维
        self.fc3 = nn.Linear(64, output_dim)  # 输出层，生成动作 logits

    def forward(self, x):
        """前向传播"""
        x = F.relu(self.fc1(x))   # ReLU 激活
        x = F.relu(self.fc2(x))   # ReLU 激活
        return self.fc3(x)        # 输出 logits，后续通过 Gumbel-Softmax 处理

# --- Critic 网络：价值网络，用于评估动作质量 ---
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents):
        """初始化 Critic 网络
        参数：
            state_dim: 每个智能体的状态维度
            action_dim: 每个智能体的动作维度
            n_agents: 智能体数量
        """
        super(Critic, self).__init__()
        input_dim = state_dim * n_agents + action_dim * n_agents  # 输入为全局状态和所有动作的拼接
        self.fc1 = nn.Linear(input_dim, 128)  # 第一层全连接
        self.fc2 = nn.Linear(128, 64)         # 第二层全连接
        self.fc3 = nn.Linear(64, 1)           # 输出层，生成 Q 值

    def forward(self, states, actions):
        """前向传播
        参数：
            states: 全局状态
            actions: 所有智能体的动作
        """
        x = torch.cat([states, actions], dim=-1)  # 拼接状态和动作
        x = F.relu(self.fc1(x))   # ReLU 激活
        x = F.relu(self.fc2(x))   # ReLU 激活
        return self.fc3(x)        # 输出 Q 值

# --- 经验回放缓冲区：存储和采样经验 ---
class ReplayBuffer:
    def __init__(self, capacity):
        """初始化回放缓冲区
        参数：
            capacity: 缓冲区最大容量
        """
        self.buffer = deque(maxlen=capacity)  # 双端队列，超出容量时自动移除旧数据

    def push(self, obs, all_obs, action, all_action, reward, all_reward, next_obs, all_next_obs, done):
        """存储一条经验"""
        self.buffer.append((obs, all_obs, action, all_action, reward, all_reward, next_obs, all_next_obs, done))

    def sample(self, batch_size):
        """随机采样一批经验
        参数：
            batch_size: 采样数量
        """
        states, all_states, actions, all_actions, rewards, all_rewards, next_states,all_next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (np.array(states), np.array(all_states), np.array(actions), np.array(all_actions), np.array(rewards), np.array(all_rewards), np.array(next_states), np.array(all_next_states), np.array(dones))

    def __len__(self):
        """返回缓冲区当前大小"""
        return len(self.buffer)

# --- MADDPG 智能体：实现多智能体强化学习 ---
class MADDPGAgent:
    def __init__(self, obs_dim, act_dim, state_dim, n_agents, actor_lr=1e-4, critic_lr=1e-3, gamma=0.95, tau=0.01, capacity=10000):
        """初始化 MADDPG 智能体
        参数：
            obs_dim: 观测维度
            act_dim: 动作维度
            state_dim: 状态维度
            n_agents: 智能体数量
            actor_lr: Actor 学习率
            critic_lr: Critic 学习率
            gamma: 折扣因子
            tau: 目标网络软更新参数
            capacity: 回放缓冲区容量
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau

        # 初始化 Actor 和 Critic 网络及其目标网络
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.target_actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(state_dim, act_dim, n_agents).to(device)
        self.target_critic = Critic(state_dim, act_dim, n_agents).to(device)

        # 复制初始参数到目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 设置优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 初始化经验回放缓冲区
        self.memory = ReplayBuffer(capacity)
        self.agents = None  # 用于共享所有智能体信息，稍后设置

    def act(self, obs, explore=True):
        """选择动作
        参数：
            obs: 当前观测
            explore: 是否进行探索（True 时使用 Gumbel-Softmax 采样）
        """
        obs = torch.FloatTensor(obs).unsqueeze(0).to(device)  # 转换为张量并增加批次维度
        with torch.no_grad():  # 不计算梯度
            logits = self.actor(obs)  # 获取 Actor 输出的 logits
            if explore:
                # 探索模式：使用 Gumbel-Softmax 采样生成 one-hot 动作
                action_probs = gumbel_softmax(logits, temperature=1.0, hard=True)
            else:
                # 评估模式：直接取最大概率动作
                action_probs = F.softmax(logits, dim=-1)
                action_probs = torch.zeros_like(action_probs).scatter_(-1, torch.argmax(action_probs, dim=-1, keepdim=True), 1.0)
        return action_probs.squeeze(0).cpu().numpy()  # 返回动作概率分布

    def store(self, obs, all_obs, action, all_action, reward, all_reward, next_obs, all_next_obs, done):
        """存储经验到回放缓冲区"""
        self.memory.push(obs, all_obs, action, all_action, reward, all_reward, next_obs, all_next_obs, done)

    def train(self, batch_size):
        """训练智能体
        参数：
            batch_size: 采样批次大小
        """
        if len(self.memory) < batch_size:  # 缓冲区数据不足时跳过
            return

        # 从回放缓冲区采样
        states, all_states, actions, all_actions, rewards, all_rewards, next_states,all_next_states, dones = self.memory.sample(batch_size)
        states = torch.FloatTensor(states).to(device)
        all_states = torch.FloatTensor(all_states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        all_actions = torch.FloatTensor(all_actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(device)
        all_rewards = torch.FloatTensor(all_rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        all_next_states = torch.FloatTensor(all_next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(device)


        # 更新 Critic
        all_next_actions = []
        for i, agent in enumerate(self.agents):
            curruent_next_action = agent.target_actor(all_next_states[:,i,:]).detach()
            all_next_actions.append(curruent_next_action)
        #矩阵展平为[batch_size,agent_num*action_dim]和[batch_size,agent_num*state_dim]
        all_next_actions= torch.cat(all_next_actions, dim=-1)
        all_states_reshape = torch.flatten(all_states, start_dim=1)
        all_next_states_reshape = torch.flatten(all_next_states, start_dim=1)
        all_actions_reshape = torch.flatten(all_actions, start_dim=1)
        self.critic_optimizer.zero_grad()  # 清零梯度
        target_q = rewards + self.gamma * self.target_critic(all_next_states_reshape, all_next_actions) * (1 - dones)  # TD 目标
        current_q = self.critic(all_states_reshape, all_actions_reshape)  # 当前 Q 值
        critic_loss = F.mse_loss(current_q, target_q.detach())  # 计算均方误差损失
        critic_loss.backward()  # 反向传播
        self.critic_optimizer.step()  # 更新参数

        # 更新 Actor
        self.actor_optimizer.zero_grad()

        all_actor_actions = []
        for i, agent in enumerate(self.agents):
            current_action = agent.actor(all_states[:,i,:])  # 当前智能体的动作
            current_action = gumbel_softmax(current_action, hard=True)
            all_actor_actions.append(current_action)
        all_actor_actions = torch.cat(all_actor_actions, dim=-1)
        actor_loss = -self.critic(all_states_reshape, all_actor_actions).mean()  # Actor 损失：最大化 Q 值
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_target(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filepath):
        """保存模型参数"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """加载模型参数"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Model loaded from {filepath}")
