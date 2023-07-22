import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout
from PyQt5.QtGui import QPainter, QColor
from PyQt5 import QtCore
import random

class Environment:
    def __init__(self):
        self.start = (3,0) # 起点
        self.end = (9,8) # 终点
        self.grid_height = 10 # 地图高度
        self.grid_width = 10 # 地图宽度
        self.max_episode_steps = 100 # 最大步数
        

    def reset(self):
        self.current_pos = self.start # 重置当前位置
        self.step_count = 0 # 步数清零
        return self.current_pos

    def step(self, action):
        self.step_count += 1
        next_pos = self.get_next_pos(action)
        reward = self.get_reward(next_pos)
        done = self.is_done(next_pos)
        self.current_pos = next_pos
        return next_pos, reward, done

    def get_next_pos(self, action):
        x, y = self.current_pos
        if action == 0: # 上
            y -= 1
        elif action == 1: # 下
            y += 1
        elif action == 2: # 左
            x -= 1
        elif action == 3: # 右
            x += 1
        return (x, y)

    def get_reward(self, next_pos):
        if next_pos == self.end:
            return 1 # 到达终点得到奖励
        else:
            return -0.1 # 否则得到负的奖励

    def is_done(self, next_pos):
        if next_pos == self.end or self.step_count >= self.max_episode_steps:
            return True
        else:
            return False

class Agent:
    def __init__(self, env):
        self.env = env
        self.alpha = 1 # 学习率
        self.discount_factor = 0.5 # 折扣因子
        self.epsilon = 1 # 探索率
        self.q_table = {} # Q表

        for i in range(env.grid_width):
            for j in range(env.grid_height):
                self.q_table[(i,j)] = [0, 0, 0, 0] # 初始化Q表，4个动作都为0

    def train(self):
        num_episodes = 10000 # 需要训练的总次数

        for episode in range(num_episodes):
            state = self.env.reset() # 获取起始状态
            done = False # 游戏结束标志位
            while not done:
                action = self.get_action(state) # 根据当前状态选择动作
                next_state, reward, done = self.env.step(action) # 获取下一个状态、奖励和游戏是否结束

                # 更新Q值
                current_q = self.q_table[state][action]
                if next_state not in self.q_table:
                    self.q_table[next_state] = [0, 0, 0, 0]
                next_max_q = max(self.q_table[next_state])
                new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.discount_factor * next_max_q)
                self.q_table[state][action] = new_q

                state = next_state # 更新状态

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon: # 探索
            action = random.randint(0, 3)
        else: # 利用
            action = self.get_max_q_action(state)
        return action

    def get_max_q_action(self, state):
        max_q = max(self.q_table[state])
        actions = []
        for i in range(4):
            if self.q_table[state][i] == max_q:
                actions.append(i)
        return random.choice(actions)

    def get_policy(self):
        policy = {}
        for i in range(self.env.grid_width):
            for j in range(self.env.grid_height):
                state = (i,j)
                if state == self.env.end:
                    continue
                max_q = max(self.q_table[state])
                actions = []
                for k in range(4):
                    if self.q_table[state][k] == max_q:
                        actions.append(k)
                policy[state] = actions
        return policy

class Car(QWidget):
    def __init__(self, size, square_size, env, agent):
        super().__init__()
        self.size = size
        self.square_size = square_size
        self.env = env
        self.agent = agent
        self.current_pos = None
        self.setGeometry(0, 0, size, size)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.run)
        self.timer.start(1)  # 每0.01秒触发一次定时器
    
    def reset(self):
        self.current_pos = self.env.reset()

    def paintEvent(self, event):
        qp = QPainter(self)

        # Draw map
        qp.setPen(QColor(0, 0, 0))
        for i in range(self.env.grid_width):
            for j in range(self.env.grid_height):
                x = i * self.square_size
                y = j * self.square_size
                qp.drawRect(x, y, self.square_size, self.square_size)

        # Draw car
        qp.setBrush(QColor(255, 0, 0))
        qp.drawRect(self.current_pos[0] * self.square_size, self.current_pos[1] * self.square_size,
                    self.square_size, self.square_size)

    def run(self):
        for i in range(self.env.max_episode_steps):
            action = self.agent.get_max_q_action(self.current_pos)
            next_pos, reward, done = self.env.step(action)
            self.current_pos = next_pos
            self.update()
            if done:
                break

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Set size of map and squares
    map_size = 500
    square_size = map_size // 10
    
    # Create window and layout
    window = QWidget()
    layout = QHBoxLayout(window)
    
    # Create car and add to layout
    env = Environment()
    agent = Agent(env)
    agent.train() 
    car = Car(map_size, square_size, env, agent)
    car.reset()
    layout.addWidget(car)
    
    # Set layout of window and show
    window.setLayout(layout)
    window.show()
    
    sys.exit(app.exec_())
