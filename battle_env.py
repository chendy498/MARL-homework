import copy
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

logger = logging.getLogger(__name__)

plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定宋体
# 定义环境的常量
CELL_SIZE = 15  # 单元格大小
WALL_COLOR = 'black'  # 墙颜色
AGENT_COLOR = 'red'  # 代理颜色
OPPONENT_COLOR = 'blue'  # 对手颜色

ACTION_MEANING = {  # 动作含义
    0: "向下",
    1: "向左",
    2: "向上",
    3: "向右",
    4: "无操作",
}

PRE_IDS = {  # 预定义标识
    'wall': 'W',  # 墙
    'empty': 'E',  # 空
    'agent': 'A',  # 代理
    'opponent': 'X',  # 对手
}

class Battle_Env:
    """
    一个简单的多代理战斗模拟环境，在一个 n x n 的网格中有两支队伍：代理和对手。
    每支队伍有 m 个代理（默认5个），初始位置在围绕随机队伍中心的 5x5 区域内。
    代理可以向四个方向移动、通过对手 ID 攻击对手，或无操作。
    如果目标在 3x3 射程范围内，攻击会将目标生命值减1，并有1步冷却时间。
    初始生命值为3，生命值为0时死亡。回合在以下情况结束：一支队伍获胜（对手全死）或100步后平局。
    对手遵循硬编码策略。

    观测：每个代理周围 5x5 网格，包含6个通道（类型、ID、健康、冷却、x坐标、y坐标）。
    动作：整数 0-4 表示移动/无操作，5 到 5+n_opponents-1 表示攻击对手。
    奖励：击中对手+1，被击中-1，可选每步成本（默认0）。
    """
    def __init__(self, grid_shape=(20, 20), n_agents=5, n_opponents=5, init_health=3, full_observable=False,step_cost=0, max_steps=100):
        self._grid_shape = grid_shape  # 网格形状
        self.n_agents = n_agents  # 代理数量
        self._n_opponents = n_opponents  # 对手数量
        self._max_steps = max_steps  # 最大步数
        self._step_cost = step_cost  # 每步成本
        self._step_count = None  # 当前步数
        # self.animate = animate  # 是否展示动画

        self.agent_pos = {i: [] for i in range(self.n_agents)}  # 代理当前位置
        self.agent_prev_pos = {i: [] for i in range(self.n_agents)}  # 代理前一位置
        self.opp_pos = {i: [] for i in range(self._n_opponents)}  # 对手当前位置
        self.opp_prev_pos = {i: [] for i in range(self._n_opponents)}  # 对手前一位置

        self._init_health = init_health  # 初始生命值
        self.agent_health = {i: 0 for i in range(self.n_agents)}  # 代理生命值
        self.opp_health = {i: 0 for i in range(self._n_opponents)}  # 对手生命值
        self._agent_cool = {i: False for i in range(self.n_agents)}  # 代理冷却状态
        self._opp_cool = {i: False for i in range(self._n_opponents)}  # 对手冷却状态
        self._total_episode_reward = None  # 总回合奖励
        self.full_observable = full_observable  # 是否完全可观测

        # 用于动画的变量
        self.fig = None
        self.ax = None
        self.history = []  # 保存每一步的状态用于动画

    def get_action_meanings(self, agent_i=None):
        """返回所有代理或特定代理的动作描述列表"""
        action_meaning = [ACTION_MEANING[i] for i in range(5)] + \
                         [f"攻击对手 {o}" for o in range(self._n_opponents)]
        if agent_i is not None:
            return action_meaning
        return [action_meaning for _ in range(self.n_agents)]

    def get_agent_obs(self):
        """返回每个代理的观测列表：5x5x6 展平的数组，后续可以改成5x5x2+4的数组"""
        _obs = []
        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]
            _agent_i_obs = np.zeros((6, 5, 5))
            for row in range(5):
                for col in range(5):
                    grid_pos = [row + (pos[0] - 2), col + (pos[1] - 2)]
                    if self.is_valid(grid_pos) and PRE_IDS['empty'] not in self._full_obs[grid_pos[0]][grid_pos[1]]:
                        x = self._full_obs[grid_pos[0]][grid_pos[1]]
                        _type = 1 if PRE_IDS['agent'] in x else -1
                        _id = int(x[1:]) - 1
                        _agent_i_obs[0][row][col] = _type  # 类型
                        _agent_i_obs[1][row][col] = _id  # ID
                        _agent_i_obs[2][row][col] = self.agent_health[_id] if _type == 1 else self.opp_health[_id]  # 健康
                        _agent_i_obs[3][row][col] = 1 if (self._agent_cool[_id] if _type == 1 else self._opp_cool[_id]) else -1  # 冷却
                        _agent_i_obs[4][row][col] = pos[0] / self._grid_shape[0]  # x坐标
                        _agent_i_obs[5][row][col] = pos[1] / self._grid_shape[1]  # y坐标
            _obs.append(_agent_i_obs.flatten().tolist())#将6x5x5的数组展平
        return _obs #长度为 150 的列表

    def __create_grid(self):
        """创建指定形状的空网格"""
        return [[PRE_IDS['empty'] for _ in range(self._grid_shape[1])] for _ in range(self._grid_shape[0])]

    def __update_agent_view(self, agent_i):
        """更新网格中代理的新位置,将原位置变成空，将当前位置变为智能体的id"""
        self._full_obs[self.agent_prev_pos[agent_i][0]][self.agent_prev_pos[agent_i][1]] = PRE_IDS['empty']
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1)

    def __update_opp_view(self, opp_i):
        """更新网格中对手的新位置"""
        self._full_obs[self.opp_prev_pos[opp_i][0]][self.opp_prev_pos[opp_i][1]] = PRE_IDS['empty']
        self._full_obs[self.opp_pos[opp_i][0]][self.opp_pos[opp_i][1]] = PRE_IDS['opponent'] + str(opp_i + 1)

    def __init_full_obs(self):
        """初始化网格，将代理和对手随机放置在区域内"""
        # 创建一个空的网格，尺寸为 self._grid_shape
        self._full_obs = self.__create_grid()

        # 随机选择代理团队的起始角落（左上角），限制在网格的前半部分
        agent_team_corner = (random.randint(0, int(self._grid_shape[0] / 2)),
                             random.randint(0, int(self._grid_shape[1] / 2)))

        # 从 5x5 区域（共 25 个位置）中随机选择 n_agents 个不重复的位置索引
        agent_pos_index = random.sample(range(25), self.n_agents)

        # 为每个代理分配初始位置
        for agent_i in range(self.n_agents):
            # 根据索引计算在 5x5 区域内的相对位置，并加上团队角落的偏移
            pos = [int(agent_pos_index[agent_i] / 5) + agent_team_corner[0],
                   agent_pos_index[agent_i] % 5 + agent_team_corner[1]]

            # 如果位置已被占用，则在 5x5 区域内重新随机选择，直到找到空位
            while self._full_obs[pos[0]][pos[1]] != PRE_IDS['empty']:
                pos = [random.randint(agent_team_corner[0], agent_team_corner[0] + 4),
                       random.randint(agent_team_corner[1], agent_team_corner[1] + 4)]

            # 设置代理的前一位置和当前位置，并更新网格视图
            self.agent_prev_pos[agent_i] = pos
            self.agent_pos[agent_i] = pos
            self.__update_agent_view(agent_i)

        """和智能体构建的代码几乎一样"""
        # 随机选择对手团队的起始角落（左上角），避免与代理重叠
        while True:
            pos = (random.randint(agent_team_corner[0], self._grid_shape[0] - 5),
                   random.randint(agent_team_corner[1], self._grid_shape[1] - 5))
            if self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty']:
                opp_team_corner = pos
                break

        # 从 5x5 区域中随机选择 n_opponents 个不重复的位置索引
        opp_pos_index = random.sample(range(25), self._n_opponents)

        # 为每个对手分配初始位置
        for opp_i in range(self._n_opponents):
            # 根据索引计算在 5x5 区域内的相对位置，并加上团队角落的偏移
            pos = [int(opp_pos_index[opp_i] / 5) + opp_team_corner[0],
                   opp_pos_index[opp_i] % 5 + opp_team_corner[1]]

            # 如果位置已被占用，则在 5x5 区域内重新随机选择，直到找到空位
            while self._full_obs[pos[0]][pos[1]] != PRE_IDS['empty']:
                pos = [random.randint(opp_team_corner[0], opp_team_corner[0] + 4),
                       random.randint(opp_team_corner[1], opp_team_corner[1] + 4)]

            # 设置对手的前一位置和当前位置，并更新网格视图
            self.opp_prev_pos[opp_i] = pos
            self.opp_pos[opp_i] = pos
            self.__update_opp_view(opp_i)

    def reset(self):
        """重置环境并返回初始观测"""
        self._step_count = 0
        #初始化幕的总奖励
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        #初始化代理和对手的生命值
        self.agent_health = {i: self._init_health for i in range(self.n_agents)}
        self.opp_health = {i: self._init_health for i in range(self._n_opponents)}
        #初始化代理和对手的冷却状态
        self._agent_cool = {i: True for i in range(self.n_agents)}
        self._opp_cool = {i: True for i in range(self._n_opponents)}
        #初始化网格，并初始化代理和对手的位置
        self.__init_full_obs()
        self.history = [self._get_state()]  # 初始化动画历史
        return self.get_agent_obs()

    def _get_state(self):
        """获取当前状态快照，用于动画"""
        return {
            'agent_pos': copy.deepcopy(self.agent_pos),
            'opp_pos': copy.deepcopy(self.opp_pos),
            'agent_health': copy.deepcopy(self.agent_health),
            'opp_health': copy.deepcopy(self.opp_health)
        }

    def render(self, mode='text'):
        """渲染环境，可选择文本或动画模式"""
        if mode == 'text' :
            grid_str = ""
            for row in self._full_obs:
                for cell in row:
                    if cell == PRE_IDS['empty']:
                        grid_str += ". "
                    elif cell.startswith(PRE_IDS['agent']):
                        grid_str += "A" + cell[1] + " "
                    elif cell.startswith(PRE_IDS['opponent']):
                        grid_str += "X" + cell[1] + " "
                    else:
                        grid_str += "? "
                grid_str += "\n"
            print(grid_str)
        elif mode == 'animate' :
            if self.fig is None:
                self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.clear()
            self.ax.set_xlim(0, self._grid_shape[1])
            self.ax.set_ylim(0, self._grid_shape[0])
            self.ax.set_xticks(range(self._grid_shape[1] + 1))
            self.ax.set_yticks(range(self._grid_shape[0] + 1))
            self.ax.grid(True)

            # 绘制代理
            for agent_i, pos in self.agent_pos.items():
                if self.agent_health[agent_i] > 0:
                    self.ax.plot(pos[1] + 0.5, self._grid_shape[0] - pos[0] - 0.5, 'o',
                                 color=AGENT_COLOR, markersize=10)
                    self.ax.text(pos[1] + 0.5, self._grid_shape[0] - pos[0] - 0.5, str(agent_i + 1),
                                 color='white', ha='center', va='center')

            # 绘制对手
            for opp_i, pos in self.opp_pos.items():
                if self.opp_health[opp_i] > 0:
                    self.ax.plot(pos[1] + 0.5, self._grid_shape[0] - pos[0] - 0.5, 'o',
                                 color=OPPONENT_COLOR, markersize=10)
                    self.ax.text(pos[1] + 0.5, self._grid_shape[0] - pos[0] - 0.5, str(opp_i + 1),
                                 color='white', ha='center', va='center')

            plt.title(f"步数: {self._step_count}")
            plt.draw()
            plt.pause(0.1)  # 短暂暂停以显示当前帧

    def show_animation(self):
        """展示整个过程的动画"""
        # if not self.animate or not self.history:
        #     print("动画未启用或无历史记录")
        #     return

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(0, self._grid_shape[1])
        self.ax.set_ylim(0, self._grid_shape[0])
        self.ax.set_xticks(range(self._grid_shape[1] + 1))
        self.ax.set_yticks(range(self._grid_shape[0] + 1))
        self.ax.grid(True)

        def update(frame):
            self.ax.clear()
            self.ax.set_xlim(0, self._grid_shape[1])
            self.ax.set_ylim(0, self._grid_shape[0])
            self.ax.set_xticks(range(self._grid_shape[1] + 1))
            self.ax.set_yticks(range(self._grid_shape[0] + 1))
            self.ax.grid(True)

            state = self.history[frame]
            for agent_i, pos in state['agent_pos'].items():
                if state['agent_health'][agent_i] > 0:
                    self.ax.plot(pos[1] + 0.5, self._grid_shape[0] - pos[0] - 0.5, 'o',
                                 color=AGENT_COLOR, markersize=10)
                    self.ax.text(pos[1] + 0.5, self._grid_shape[0] - pos[0] - 0.5, str(agent_i + 1),
                                 color='white', ha='center', va='center')
            for opp_i, pos in state['opp_pos'].items():
                if state['opp_health'][opp_i] > 0:
                    self.ax.plot(pos[1] + 0.5, self._grid_shape[0] - pos[0] - 0.5, 'o',
                                 color=OPPONENT_COLOR, markersize=10)
                    self.ax.text(pos[1] + 0.5, self._grid_shape[0] - pos[0] - 0.5, str(opp_i + 1),
                                 color='white', ha='center', va='center')
            self.ax.set_title(f"步数: {frame}")
            return []

        ani = FuncAnimation(self.fig, update, frames=len(self.history), interval=500, repeat=False)
        plt.show()

    def __update_agent_pos(self, agent_i, move):
        """根据移动动作更新代理位置"""
        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = None
        if move == 0:  # 向下
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # 向左
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # 向上
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # 向右
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # 无操作
            pass
        else:
            raise ValueError(f"无效动作: {move}")
        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.agent_pos[agent_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.__update_agent_view(agent_i)

    def __update_opp_pos(self, opp_i, move):
        """根据移动动作更新对手位置"""
        curr_pos = copy.copy(self.opp_pos[opp_i])
        next_pos = None
        if move == 0:  # 向下
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # 向左
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # 向上
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # 向右
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # 无操作
            pass
        else:
            raise ValueError(f"无效动作: {move}")
        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.opp_pos[opp_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.__update_opp_view(opp_i)

    def is_valid(self, pos):
        """检查位置是否在网格内"""
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _is_cell_vacant(self, pos):
        """检查位置是否为空"""
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])

    @staticmethod
    def is_visible(source_pos, target_pos):
        """检查目标是否在源位置的 5x5 视野范围内"""
        return (source_pos[0] - 2) <= target_pos[0] <= (source_pos[0] + 2) and \
               (source_pos[1] - 2) <= target_pos[1] <= (source_pos[1] + 2)

    @staticmethod
    def is_iterable(source_pos, target_pos):
        """检查目标是否在源位置的 3x3 射程范围内"""
        return (source_pos[0] - 1) <= target_pos[0] <= (source_pos[0] + 1) and \
               (source_pos[1] - 1) <= target_pos[1] <= (source_pos[1] + 1)

    def reduce_distance_move(self, source_pos, target_pos):
        """返回减少与目标距离的移动动作"""
        _moves = []
        if source_pos[0] > target_pos[0]:
            _moves.append(2)  # 向上
        elif source_pos[0] < target_pos[0]:
            _moves.append(0)  # 向下
        if source_pos[1] > target_pos[1]:
            _moves.append(1)  # 向左
        elif source_pos[1] < target_pos[1]:
            _moves.append(3)  # 向右
        return random.choice(_moves) if _moves else 4  # 无操作如果无移动

    @property
    def opps_action(self):
        """根据对手策略计算动作"""
        visible_agents = set()
        opp_agent_distance = {i: [] for i in range(self._n_opponents)}
        for opp_i, opp_pos in self.opp_pos.items():
            for agent_i, agent_pos in self.agent_pos.items():
                if self.agent_health[agent_i] > 0 and self.is_visible(opp_pos, agent_pos):
                    visible_agents.add(agent_i)
                distance = abs(agent_pos[0] - opp_pos[0]) + abs(agent_pos[1] - opp_pos[1])
                opp_agent_distance[opp_i].append([distance, agent_i])

        opp_action_n = []
        for opp_i in range(self._n_opponents):
            action = None
            for _, agent_i in sorted(opp_agent_distance[opp_i]):
                if agent_i in visible_agents:
                    if self.is_iterable(self.opp_pos[opp_i], self.agent_pos[agent_i]):
                        action = 5 + agent_i
                    else:
                        action = self.reduce_distance_move(self.opp_pos[opp_i], self.agent_pos[agent_i])
                    break
            opp_action_n.append(action if action is not None else random.choice(range(5)))
        return opp_action_n

    def step(self, agents_action):
        """处理一步代理动作，返回观测、奖励、是否结束、信息"""
        assert len(agents_action) == self.n_agents, f"预期 {self.n_agents} 个动作，收到 {len(agents_action)} 个"
        self._step_count += 1
        rewards = [self._step_cost for _ in range(self.n_agents)]
        agent_health, opp_health = copy.copy(self.agent_health), copy.copy(self.opp_health)

        # 处理代理攻击
        for agent_i, action in enumerate(agents_action):
            if self.agent_health[agent_i] > 0 and action > 4:
                target_opp = action - 5
                if 0 <= target_opp < self._n_opponents and self._agent_cool[agent_i]:
                    if self.is_iterable(self.agent_pos[agent_i], self.opp_pos[target_opp]) and opp_health[target_opp] > 0:
                        opp_health[target_opp] -= 1
                        rewards[agent_i] += 1
                        self._agent_cool[agent_i] = False
                        if opp_health[target_opp] == 0:
                            pos = self.opp_pos[target_opp]
                            self._full_obs[pos[0]][pos[1]] = PRE_IDS['empty']

        # 处理对手攻击
        opp_action = self.opps_action
        for opp_i, action in enumerate(opp_action):
            if self.opp_health[opp_i] > 0 and action > 4 and self._opp_cool[opp_i]:
                target_agent = action - 5
                if 0 <= target_agent < self.n_agents:
                    if self.is_iterable(self.opp_pos[opp_i], self.agent_pos[target_agent]) and agent_health[target_agent] > 0:
                        agent_health[target_agent] -= 1
                        rewards[target_agent] -= 1
                        self._opp_cool[opp_i] = False
                        if agent_health[target_agent] == 0:
                            pos = self.agent_pos[target_agent]
                            self._full_obs[pos[0]][pos[1]] = PRE_IDS['empty']

        self.agent_health, self.opp_health = agent_health, opp_health

        # 处理移动并重置冷却
        for agent_i, action in enumerate(agents_action):
            if self.agent_health[agent_i] > 0:
                if action <= 4:
                    self.__update_agent_pos(agent_i, action)
                if not self._agent_cool[agent_i]:
                    self._agent_cool[agent_i] = True

        for opp_i, action in enumerate(opp_action):
            if self.opp_health[opp_i] > 0:
                if action <= 4:
                    self.__update_opp_pos(opp_i, action)
                if not self._opp_cool[opp_i]:
                    self._opp_cool[opp_i] = True

        # 检查是否结束
        done = (self._step_count >= self._max_steps or
                sum(self.opp_health.values()) == 0 or
                sum(self.agent_health.values()) == 0)
        win = sum(self.opp_health.values()) == 0

        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]

        # 保存当前状态到动画历史
        self.history.append(self._get_state())

        return self.get_agent_obs(), rewards, done, {'health': self.agent_health, 'win': win}

    def seed(self, n):
        """设置随机种子以确保可重复性"""
        random.seed(n)
        np.random.seed(n)

# 测试脚本
if __name__ == "__main__":
    # 设置环境，启用动画
    env = Battle_Env()
    obs = env.reset()
    print("初始观测长度:", len(obs), "第一个代理观测长度:", len(obs[0]))
    env.render(mode='text')  # 初始状态的文本渲染

    # 模拟几步
    for _ in range(5):
        actions = [random.randint(0, 4) for _ in range(env.n_agents)]  # 随机移动
        obs, rewards, done, info = env.step(actions)
        # env.render(mode='animate')  # 每步动画渲染
        print(f"步数 {_+1} - 奖励:", rewards, "是否结束:", done)
        if done:
            break

    # 展示完整动画
    # env.show_animation()