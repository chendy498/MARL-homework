import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from battle_env import Battle_Env  # 导入环境
from maddpg_agent import MADDPGAgent  # 导入MADDPG智能体
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练器
class RedAgentTrainer:
    def __init__(self, algorithm='iddqn', n_agents=5, train=True):
        self.env = Battle_Env(n_agents=n_agents)
        self.n_agents = n_agents
        self.algorithm = algorithm.lower() #
        self.train = train
        self.obs_dim = 150  # 5x5x6
        self.act_dim = 5 + n_agents  # 移动 + 攻击动作
        self.state_dim = self.obs_dim  # 简化假设

        if self.algorithm == 'maddpg':
            self.agents = [MADDPGAgent(self.obs_dim, self.act_dim, self.state_dim, n_agents) for _ in range(n_agents)]
            for agent in self.agents:
                agent.agents = self.agents  # 共享所有代理信息
        elif self.algorithm == 'qmix':
            raise NotImplementedError("QMIX 未完全实现，可扩展此处")
        else:
            raise ValueError(f"不支持的算法: {algorithm}")

    def run_episode(self, max_steps=100):
        obs = self.env.reset()#重置环境并返回初始观察
        total_rewards = [0] * self.n_agents
        all_obs, all_actions, all_next_obs = [], [], []

        for step in range(max_steps):
            actions = [agent.act(obs[i]) for i, agent in enumerate(self.agents)]
            next_obs, rewards, done, info = self.env.step(actions)
            all_obs.append(obs)
            all_actions.append(actions)
            all_next_obs.append(next_obs)

            # if self.train:
            #     for i, agent in enumerate(self.agents):
            #         agent.store(obs[i], actions[i], rewards[i], next_obs[i], done)
            #         if self.algorithm == 'iddqn':
            #             agent.train()
            #         elif self.algorithm == 'maddpg':
            #             agent.train(batch_size=64, all_obs=np.vstack(all_obs[-1]),
            #                         all_actions=np.vstack(all_actions[-1]),
            #                         all_next_obs=np.vstack(all_next_obs[-1]))
            if self.train:
                for i, agent in enumerate(self.agents):
                     agent.store(obs[i], actions[i], rewards[i], next_obs[i], done)
            if self.algorithm == 'madddpg':
                self.train_MADDPGAgent()
            elif self.algorithm == 'qmix':
                self.train_QMIXAgent()
            elif self.algorithm =='mappo':
                self.train_MAPPOAgent()

            for i in range(self.n_agents):
                total_rewards[i] += rewards[i]
            obs = next_obs


            if not self.train:
                self.env.render(mode='animate')#动画展示

            if done:
                break

        if self.train and self.algorithm == 'maddpg':
            for agent in self.agents:
                agent.update_target()

        if not self.train:
            self.env.show_animation()#动画展示

        return total_rewards

    def train_agents(self, episodes=100):
        if not self.train:
            print("训练未启用，仅运行单回合")
            rewards = self.run_episode()
            print(f"单回合奖励: {rewards}")
            return

        for ep in range(episodes):
            rewards = self.run_episode()
            print(f"回合 {ep + 1}/{episodes} - 总奖励: {sum(rewards)}")

    def train_MADDPGAgent(self):
        pass
    def train_QMIXAgent(self):
        pass
    def train_MAPPOAgent(self):
        pass

if __name__ == "__main__":
    # 测试不同算法和训练开关
    # trainer = RedAgentTrainer(algorithm='iddqn', train=True)
    # trainer.train_agents(episodes=50)

    trainer = RedAgentTrainer(algorithm='maddpg')
    trainer.train_agents(episodes=10)

    #SAC，DQN，PPO，
    # QMIX，MADDPG 最初为连续动作空间设计，但可以通过离散化技术