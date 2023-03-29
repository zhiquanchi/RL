import sys
import numpy as np
import random
import time

from PyQt5 import QtGui, QtCore, QtWidgets

# 定义窗口类
class MapWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.title = '小车寻路'
        self.width = 800
        self.height = 800
        self.initUI()

    # 初始化UI
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(0, 0, self.width, self.height)

        # 定义按钮
        self.start_button = QtWidgets.QPushButton('开始训练', self)
        self.start_button.clicked.connect(self.start_training)
        self.start_button.move(20, 20)
        self.stop_button = QtWidgets.QPushButton('停止训练', self)
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.move(120, 20)
        self.reset_button = QtWidgets.QPushButton('重置环境', self)
        self.reset_button.clicked.connect(self.reset_environment)
        self.reset_button.move(220, 20)

        self.show()

    # 开始训练
    def start_training(self):
        # 创建Q-table
        self.Q = np.zeros([self.state_num, self.action_num])

        # 开始循环训练
        for episode in range(self.max_episode):
            state = self.env.reset()
            r_sum = 0

            for step in range(self.max_step):
                # 根据当前状态选择动作
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.Q[state, :])

                # 执行动作
                next_state, reward, done, _ = self.env.step(action)

                # 更新Q值
                self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state, :]))

                state = next_state
                r_sum += reward

                # 如果到达终点，则重置环境
                if done:
                    break

            # 降低epsilon和alpha值，提高收敛效果
            self.epsilon = self.epsilon * self.epsilon_decay
            self.alpha = self.alpha * self.alpha_decay

            # 实时更新GUI，并显示当前状态和奖励
            self.update_gui(state, r_sum)

            # 如果达到终止条件，则停止训练
            if r_sum >= self.max_reward:
                break

    # 实时更新GUI
    def update_gui(self, state, r_sum):
        self.pixmap.fill(QtGui.QColor(255, 255, 255))

        # 绘制地图
        for i in range(self.map_width):
            for j in range(self.map_height):
                if self.env.map[i][j] == '#':
                    self.painter.fillRect(i * self.grid_size, j * self.grid_size, self.grid_size, self.grid_size, QtGui.QColor(0, 0, 0))
                elif self.env.map[i][j] == 'S':
                    self.painter.fillRect(i * self.grid_size, j * self.grid_size, self.grid_size, self.grid_size, QtGui.QColor(0, 255, 0))
                elif self.env.map[i][j] == 'G':
                    self.painter.fillRect(i * self.grid_size, j * self.grid_size, self.grid_size, self.grid_size, QtGui.QColor(255, 0, 0))

        # 绘制小车
        x, y = self.env.get_position(state)
        self.painter.fillRect(x * self.grid_size, y * self.grid_size, self.grid_size, self.grid_size, QtGui.QColor(0, 0, 255))

        # 实时显示状态和奖励
        state_label = QtWidgets.QLabel('当前状态：' + str(state))
        state_label.move(20, 100)
        state_label.setParent(self)
        reward_label = QtWidgets.QLabel('累计奖励：' + str(r_sum))
        reward_label.move(20, 120)
        reward_label.setParent(self)

        self.update()

    # 停止训练
    def stop_training(self):
        self.is_running = False

    # 重置环境
    def reset_environment(self):
        self.env.reset()
        self.Q = np.zeros([self.state_num, self.action_num])
        self.update_gui(0, 0)

    # 绘制地图
    def paintEvent(self, event):
        self.painter = QtGui.QPainter(self)
        self.painter.drawPixmap(0, 0,self.Qpixmap) # DEBUG

    # 初始化环境
    def init_environment(self):
        self.env = Map(self.map_width, self.map_height, self.grid_size)

        self.state_num = self.env.width * self.env.height
        self.action_num = len(self.env.action_space)

    def run(self):
        self.is_running = True

        # 设置超参数
        self.epsilon = 0.9
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon_decay = 0.99
        self.alpha_decay = 0.99
        self.max_episode = 10000
        self.max_step = 1000
        self.max_reward = float('inf')

        # 初始化环境
        self.init_environment()

        # 创建Q-table
        self.Q = np.zeros([self.state_num, self.action_num])

        # 初始化GUI
        self.pixmap = QtGui.QPixmap(self.width, self.height)
        self.pixmap.fill(QtGui.QColor(255, 255, 255))

        # 开始循环训练
        for episode in range(self.max_episode):
            state = self.env.reset()
            r_sum = 0

            for step in range(self.max_step):
                # 根据当前状态选择动作
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.Q[state, :])

                # 执行动作
                next_state, reward, done, _ = self.env.step(action)

                # 更新Q值
                self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state, :]))

                state = next_state
                r_sum += reward

                # 如果到达终点，则重置环境
                if done:
                    break

            # 降低epsilon和alpha值，提高收敛效果
            self.epsilon = self.epsilon * self.epsilon_decay
            self.alpha = self.alpha * self.alpha_decay

            # 实时更新GUI，并显示当前状态和奖励
            self.update_gui(state, r_sum)

            # 如果达到终止条件，则停止训练
            if r_sum >= self.max_reward:
                break

            # 实时刷新页面
            QtWidgets.QApplication.processEvents()

            # 如果停止按钮被按下，则停止训练
            if not self.is_running:
                break

        self.update()

# 定义小车所在的地图类
class Map:
    def __init__(self, width, height, grid_size):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.action_space = [0, 1, 2, 3] # 上下左右
        self.map = [
            ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#'],
            ['#', 'S', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
            ['#', '#', '#', '#', '#', '#', ' ', ' ', ' ', '#'],
            ['#', ' ', '#', ' ', ' ', ' ', ' ', '#', ' ', '#'],
            ['#', ' ', ' ', '#', '#', '#', '#', '#', ' ', '#'],
            ['#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
            ['#', ' ', '#', '#', '#', '#', '#', '#', '#', '#'],
            ['#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'G', '#'],
            ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#']
        ]
        self.reset()

    def reset(self):
        self.position = [1, 1]
        self.state = self.get_state()
        return self.state

    def step(self, action):
        if action == 0: # 向上移动
            next_position = [self.position[0], self.position[1] - 1]
        elif action == 1: # 向下移动
            next_position = [self.position[0], self.position[1] + 1]
        elif action == 2: # 向左移动
            next_position = [self.position[0] - 1, self.position[1]]
        elif action == 3: # 向右移动
            next_position = [self.position[0] + 1, self.position[1]]

        # 判断下一步是否越界或者碰到障碍物
        if next_position[0] < 0 or next_position[0] >= self.width or next_position[1] < 0 or next_position[1] >= self.height or self.map[next_position[0]][next_position[1]] == '#':
            reward = -10
            next_position = self.position
        elif self.map[next_position[0]][next_position[1]] == 'G': # 到达终点
            reward = 100
            done = True
        else:
            reward = -1

        self.position = next_position
        self.state = self.get_state()
        return self.state, reward, done, {}

    def get_state(self):
        return self.position[1] * self.width + self.position[0]

    def get_position(self, state):
        x = state % self.width
        y = int((state - x) / self.width)
        return x, y

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    map_window = MapWindow()
    map_window.run()
    sys.exit(app.exec_())
