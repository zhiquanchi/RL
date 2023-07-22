import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsRectItem
from PyQt5.QtCore import QTimer,Qt

class Maze(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create a scene
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, 500, 500)

        # Create a view
        self.view = QGraphicsView(self.scene, self)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setFixedSize(500, 500)

        # Add a rectangle to the scene
        pen = Qt.black
        brush = Qt.gray
        rect = QGraphicsRectItem(0, 0, 50, 50)
        rect.setPen(pen)
        rect.setBrush(brush)
        self.scene.addItem(rect)
        
        # 创建10*10的地图环境
        for i in range(10):
            for j in range(10):
                j = QGraphicsRectItem(i*50,j*50,50,50)
                self.scene.addItem(j)


        self.setCentralWidget(self.view)
        self.setWindowTitle('Maze')
        self.show()

    #q-table强化学习算法
    def agent_move(self):
        #创建q-table
        self.q_table = np.zeros((10, 10, 5))  # Q-table，大小为10x10x5
        self.alpha = 1  # 学习率
        self.gamma = 1 # 折扣因子
        self.epsilon = 0.6  # 探索概率
        # 活动策略
        action = np.array([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]])  # 上右下左停
        # 创建地图
        map = np.zeros((10, 10))
        map[1, 1] = 1  # 障碍物
        map[1, 2] = 1  # 障碍物
        map[1, 3] = 1  # 障碍物
        map[1, 4] = 1  # 障碍物
        map[1, 5] = 1  # 障碍物
        map[1, 6] = 1  # 障碍物
        map[1, 7] = 1  # 障碍物
        #创建活动
        for i in range(1000):
            # 选取起始位置
            state = np.array([0, 0])
            # 选取动作
            if np.random.rand() < self.epsilon:
                action_index = np.random.randint(0, 5)
            else:
                action_index = np.argmax(self.q_table[state[0], state[1], :])
            # 移动
            next_state = state + action[action_index]
            # 判断是否撞墙
            if next_state[0] < 0 or next_state[0] > 9 or next_state[1] < 0 or next_state[1] > 9 or map[
                next_state[0], next_state[1]] == 1:
                next_state = state
            # 判断是否到达终点
            if next_state[0] == 9 and next_state[1] == 9:
                self.q_table[state[0], state[1], action_index] = self.q_table[state[0], state[1], action_index] + \
                                                                self.alpha * (1 - self.q_table[state[0], state[1],
                                                                                               action_index])
            else:
                self.q_table[state[0], state[1], action_index] = self.q_table[state[0], state[1], action_index] + \
                                                                self.alpha * (
                                                                            0 + self.gamma * np.max(
                                                                        self.q_table[next_state[0], next_state[1], :]) -
                                                                            self.q_table[state[0], state[1],
                                                                                         action_index])
            state = next_state
        # 在地图上显示当前位置
        for i in range(10):
            for j in range(10):
                if np.argmax(self.q_table[i, j, :]) == 0:
                    self.scene.addLine(i * 50 + 25, j * 50 + 25, i * 50 + 25, j * 50, Qt.red)
                elif np.argmax(self.q_table[i, j, :]) == 1:
                    self.scene.addLine(i * 50 + 25, j * 50 + 25, i * 50 + 50, j * 50 + 25, Qt.red)
                elif np.argmax(self.q_table[i, j, :]) == 2:
                    self.scene.addLine(i * 50 + 25, j * 50 + 25, i * 50 + 25, j * 50 + 50, Qt.red)
                elif np.argmax(self.q_table[i, j, :]) == 3:
                    self.scene.addLine(i * 50 + 25, j * 50 + 25, i * 50, j * 50 + 25, Qt.red)
        # 奖赏策略
        reward = np.zeros((10, 10))
        reward[9, 9] = 1  # 终点位置的奖赏为1

        # 创建定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.agent_move)
        self.timer.start(10)  # 每0.01秒触发一次定时器


if __name__ == '__main__':
    app = QApplication(sys.argv)
    maze = Maze()
    sys.exit(app.exec_())

