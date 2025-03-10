import numpy as np
import torch
from battle_env import Battle_Env
from ippo_agent import IPPOAgent
from maddpg_agent import MADDPGAgent

import matplotlib.pyplot as plt
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RedAgentTrainer:
    def __init__(self, algorithm='maddpg', n_agents=5, train=True):
        """初始化训练器
        参数：
            algorithm: 使用的算法（默认 'maddpg'）
            n_agents: 智能体数量
            train: 是否进行训练
        """
        self.env = Battle_Env(n_agents=n_agents)  # 初始化环境
        self.n_agents = n_agents
        self.algorithm = algorithm.lower()
        self.train = train
        self.obs_dim = 150  # 每个智能体的观测维度
        self.act_dim = 5 + n_agents  # 动作空间：5 个移动 + n_agents 个攻击
        self.state_dim = self.obs_dim  # 简化假设：状态维度等于观测维度
        self.total_rewards = []  # 记录每个回合的总奖励
        self.all_rewards = []  # 记录每个回合的所有智能体总奖励

        if self.algorithm == 'maddpg':
            # 创建多个 MADDPG 智能体
            self.agents = [MADDPGAgent(self.obs_dim, self.act_dim, self.state_dim, n_agents) for _ in range(n_agents)]
            for agent in self.agents:
                agent.agents = self.agents  # 共享智能体列表以便协作
        elif self.algorithm =='ippo':
            self.agents = [IPPOAgent(self.obs_dim, self.act_dim, self.state_dim, n_agents) for _ in range(n_agents)]
            for agent in self.agents:
                agent.agents = self.agents
        else:
            raise ValueError(f"不支持的算法: {algorithm}")

    def run_episode(self, max_steps=100):
        """运行一个回合
        参数：
            max_steps: 最大步数
        """
        obs = self.env.reset()  # 重置环境
        total_rewards = [0] * self.n_agents  # 记录每个智能体的总奖励

        for step in range(max_steps):
            # 每个智能体选择动作
            actions = [agent.act(obs[i], explore=self.train) for i, agent in enumerate(self.agents)]
            action_indices = [np.argmax(act) for act in actions]  # 将 one-hot 动作转换为索引
            next_obs, rewards, done, info = self.env.step(action_indices)  # 执行动作
            # episode_transitions = [{'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []} for _ in range(self.n_agents)]# 用于存储每个智能体的经验

            if self.train:
                if self.algorithm == 'maddpg':
                    # 存储经验
                    for i, agent in enumerate(self.agents):
                        agent.store(obs[i],obs ,actions[i],actions, rewards[i],total_rewards, next_obs[i],next_obs, done)
                elif self.algorithm == 'ippo':
                    for i, agent in enumerate(self.agents):
                            agent.store(obs[i], actions[i], rewards[i], next_obs[i], done)# 保存状态，每次都更新
                    self.train_IPPOAgent()# 训练 IPPO 智能体，在线学习


            # 累加奖励
            for i in range(self.n_agents):
                total_rewards[i] += rewards[i]
            obs = next_obs

            if not self.train:
                self.env.render(mode='animate')  # 非训练模式下渲染动画

            if done:
                break

        if self.train and self.algorithm == 'maddpg':
            # 训练并更新目标网络
            self.train_MADDPGAgent()
            for agent in self.agents:
                agent.update_target()



        if not self.train:
            self.env.show_animation()  # 显示动画

        return total_rewards

    def train_MADDPGAgent(self):
        """训练 MADDPG 智能体"""
        for agent in self.agents:
            agent.train(batch_size=64)  # 使用批次大小 64 进行训练

    def train_IPPOAgent(self):
        """训练 IPPO 智能体"""
        for i, agent in enumerate(self.agents):
                agent.train()


    def train_agents(self, episodes=100):
        """训练多个回合
        参数：
            episodes: 回合数
        """
        if not self.train:
            print("训练未启用，仅运行单回合")
            self.load_models()
            rewards = self.run_episode()
            print(f"单回合奖励: {rewards}")
            return

        for ep in range(episodes):
            rewards = self.run_episode()
            all_rewards = sum(rewards)
            self.total_rewards.append(rewards)
            self.all_rewards.append(all_rewards)
            if ep % 100 == 0:
                print(f"回合 {ep + 1}/{episodes} - 总奖励: {all_rewards}")
        self.save_models()

    def plot_rewards(self):
        plt.figure(figsize=(10, 6))
        episodes = range(1, len(self.all_rewards) + 1)
        plt.plot(episodes, self.all_rewards, label='Raw Total Reward', color='blue', alpha=0.3)

        window_size = 100
        if len(self.all_rewards) >= window_size:
            smoothed_rewards = np.convolve(self.all_rewards, np.ones(window_size) / window_size, mode='valid')
            smoothed_episodes = episodes[window_size - 1:]
            plt.plot(smoothed_episodes, smoothed_rewards, label=f'Smoothed Reward (window={window_size})', color='red')

        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode')
        plt.grid(True)
        # plt.legend()
        plt.show()

    def save_models(self):
        """保存所有智能体的模型"""
        for i, agent in enumerate(self.agents):
            agent.save(f"{self.algorithm}_agent_{i}")

    def load_models(self):
        """加载所有智能体的模型"""
        for i, agent in enumerate(self.agents):
            agent.load(f"{self.algorithm}_agent_{i}")

if __name__ == "__main__":
    # 示例运行：训练 5 个智能体，10 个回合
    trainer = RedAgentTrainer(algorithm='ippo', n_agents=5, train=True)
    trainer.train_agents(episodes=50000)
    trainer.plot_rewards()