# ippo_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# 设置设备：如果有GPU则使用CUDA，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        """初始化策略网络（Actor）
        参数：
            input_dim: 输入维度（观测空间的大小）
            output_dim: 输出维度（动作空间的大小）
        """
        super(Actor, self).__init__()
        # 第一层全连接层，将输入维度映射到128维隐藏层
        self.fc1 = nn.Linear(input_dim, 128)
        # 第二层全连接层，将128维隐藏层映射到64维
        self.fc2 = nn.Linear(128, 64)
        # 输出层，将64维隐藏层映射到动作空间维度，输出动作的logits
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        """前向传播，生成动作的logits
        参数：
            x: 输入的观测值（张量）
        返回：
            logits: 未经过softmax的动作概率分布
        """
        x = F.relu(self.fc1(x))  # 第一层使用ReLU激活函数
        x = F.relu(self.fc2(x))  # 第二层使用ReLU激活函数
        return F.softmax(self.fc3(x), dim=1)


class Critic(nn.Module):
    def __init__(self, input_dim):
        """初始化价值网络（Critic）
        参数：
            input_dim: 输入维度（观测空间的大小）
        """
        super(Critic, self).__init__()
        # 第一层全连接层，将输入维度映射到128维隐藏层
        self.fc1 = nn.Linear(input_dim, 128)
        # 第二层全连接层，将128维隐藏层映射到64维
        self.fc2 = nn.Linear(128, 64)
        # 输出层，将64维隐藏层映射到1维，表示状态价值
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        """前向传播，计算状态价值
        参数：
            x: 输入的观测值（张量）
        返回：
            value: 状态的价值估计
        """
        x = F.relu(self.fc1(x))  # 第一层使用ReLU激活函数
        x = F.relu(self.fc2(x))  # 第二层使用ReLU激活函数
        return self.fc3(x)  # 输出状态价值，不加激活函数


class IPPOAgent:
    def __init__(self, obs_dim, act_dim, state_dim, n_agents,
                 actor_lr=3e-4,critic_lr = 1e-3, gamma=0.99,lamda=0.97, clip_param=0.2, value_loss_coef=0.5,
                 entropy_coef=0.01, max_grad_norm=0.5):
        """初始化IPPO智能体
        参数：
            obs_dim: 观测维度（每个智能体的观测空间大小）
            act_dim: 动作维度（动作空间大小）
            state_dim: 状态维度（这里与观测维度相同）
            n_agents: 智能体数量
            lr: 学习率（控制优化步长）
            gamma: 折扣因子（未来奖励的衰减率）
            clip_param: PPO的剪切参数（控制策略更新的幅度）
            value_loss_coef: 价值损失的权重系数
            entropy_coef: 熵正则化系数（鼓励探索）
            max_grad_norm: 最大梯度范数（用于梯度裁剪，防止更新过大）
            epochs: 每条序列的训练轮数
            lamda: GAE的λ参数
        """
        self.lamda = lamda  # GAE的λ参数
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.epochs=5

        self.transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}# 用于存储经验

        # 初始化网络
        self.actor = Actor(obs_dim, act_dim).to(device)  # 策略网络
        self.critic = Critic(obs_dim).to(device)  # 价值网络

        # 初始化优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)  # Actor的Adam优化器
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)  # Critic的Adam优化器

        self.agents = None  # 用于存储所有智能体的引用，由训练器设置

    def act(self, obs, explore=True):
        """选择动作
        参数：
            obs: 当前观测值（numpy数组）
            explore: 是否进行探索（True时随机采样，False时选择最优动作）
        返回：
            action: one-hot编码的动作（长度为self.act_dim的列表）
        """
        state = torch.tensor([obs], dtype=torch.float).to(device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        if explore:
            # 随机采样动作
            action = action_dist.sample()
        else:
            # 选择概率最大的动作（贪婪策略）
            action = torch.argmax(probs, dim=1)
        return action.item()


    def train(self):

        # """更新策略网络和价值网络"""
        states = torch.tensor(self.transition_dict['states'], dtype=torch.float).to(device)
        actions = torch.tensor(self.transition_dict['actions']).view(-1, 1).to(device)
        rewards = torch.tensor(self.transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(device)
        next_states = torch.tensor(self.transition_dict['next_states'], dtype=torch.float).to(device)
        dones = torch.tensor(self.transition_dict['dones'], dtype=torch.float).view(-1, 1).to(device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = self.compute_advantage(td_delta.cpu()).to(device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        log_probs = torch.log(self.actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
        actor_loss = torch.mean(-torch.min(surr1, surr2))
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def compute_advantage(self, td_delta):
        """使用 PyTorch 计算 GAE（广义优势估计）"""
        advantage_list = []
        advantage = 0.0
        for delta in reversed(td_delta):  # 直接用 PyTorch 逆序遍历
            advantage = self.gamma * self.lamda * advantage + delta  # 开销有点大，可以存储后减少计算
            advantage_list.append(advantage)

        return advantage

    def store(self, obs,  action, reward,  next_obs,  done):
        self.transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        self.transition_dict['states'].append(obs)
        self.transition_dict['actions'].append(action)
        self.transition_dict['rewards'].append(reward)
        self.transition_dict['next_states'].append(next_obs)
        self.transition_dict['dones'].append(done)


    def update_target(self):
        """占位函数，与MADDPG兼容，IPPO不使用目标网络"""
        pass

    def save(self, filepath):
        """保存 IPPO 模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """加载 IPPO 模型"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # 简单测试
    agent = IPPOAgent(obs_dim=150, act_dim=10, state_dim=150, n_agents=5)
    obs = np.random.randn(150)
    action = agent.act(obs)
    print(f"示例动作: {action}")