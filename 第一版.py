import random
import numpy as np

from PyQt5 import QtWidgets, QtGui, QtCore


class Map(QtWidgets.QGraphicsView):
    def __init__(self,start_pos,end_pos):
        super().__init__()
        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)

        # 设置场景大小
        self.scene.setSceneRect(0, 0, 100 * 100, 100 * 100)


        # 添加方格到场景中
        for i in range(10):
            for j in range(10):
                rect = QtCore.QRectF(i * 100, j * 100, 100, 100)
                item = QtWidgets.QGraphicsRectItem(rect)
                item.setPen(QtGui.QPen(QtCore.Qt.black))
                item.setBrush(QtGui.QBrush(QtCore.Qt.white))
                self.scene.addItem(item)
        
        self.start_pos = start_pos  # 起点位置
        self.end_pos = end_pos  # 终点位置

        # 将起点的方格涂成黄色
        start_rect = QtCore.QRectF(start_pos[0] * 10, start_pos[1] * 10, 100, 100)
        start_item = QtWidgets.QGraphicsRectItem(start_rect)
        start_item.setPen(QtGui.QPen(QtCore.Qt.black))
        start_item.setBrush(QtGui.QBrush(QtCore.Qt.yellow))
        self.scene.addItem(start_item)

        # 将终点的方格涂成绿色
        end_rect = QtCore.QRectF(end_pos[0] * 100, end_pos[1] * 100, 100, 100)
        end_item = QtWidgets.QGraphicsRectItem(end_rect)
        end_item.setPen(QtGui.QPen(QtCore.Qt.black))
        end_item.setBrush(QtGui.QBrush(QtCore.Qt.green))
        self.scene.addItem(end_item)
        
        # 添加小车到场景中
        car_rect = QtCore.QRectF(0, 0, 100, 100)
        self.car_item = QtWidgets.QGraphicsRectItem(car_rect)
        self.car_item.setPen(QtGui.QPen(QtCore.Qt.black))
        self.car_item.setBrush(QtGui.QBrush(QtCore.Qt.red))
        self.scene.addItem(self.car_item)
        self.car_pos = (0, 0)  # 小车的初始位置

        self.q_table = np.zeros((10, 10, 5))  # Q-table，大小为10x10x5
        self.alpha = 1  # 学习率
        self.gamma = 1 # 折扣因子
        self.epsilon = 0.6  # 探索概率

        # 创建定时器
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.agent_move)
        self.timer.start(100)  # 每0.01秒触发一次定时器

    def set_car_pos(self, x, y):
        """
        设置小车的位置
        """
        if x >= 0 and x <= 9 and y >= 0 and y <= 9:
            self.car_pos = (x, y)
            self.car_item.setPos(x * 100, y * 100)

    def agent_move(self):
        """
        代理决策并移动小车
        """
        # 获取当前状态
        state = self.get_state()

        # 根据epsilon-greedy策略选择动作
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, 4)
        else:
            action = np.argmax(self.q_table[state[0], state[1]])

        # 执行动作并获得下一个状态和奖励值
        next_state, reward = self.do_action(action)

        # 更新Q-table
        self.q_table[state[0], state[1], action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state[0], next_state[1]]) - self.q_table[state[0], state[1], action])

        # 移动小车到新的位置
        self.set_car_pos(*next_state)

    def get_state(self):
        """
        获取当前状态
        """
        return self.car_pos[0], self.car_pos[1]

    def do_action(self, action):
        """
        执行动作并返回下一个状态和奖励值
        """
        x, y = self.car_pos

        if action == 0:  # 上移
            y = max(0, y - 1)
        elif action == 1:  # 下移
            y = min(9, y + 1)
        elif action == 2:  # 左移
            x = max(0, x - 1)
        elif action == 3:  # 右移
            x = min(9, x + 1)

        # 计算奖励值
        if (x, y) == self.end_pos:
            reward = 1.0
            self.timer.stop()
            # msgbox = QtWidgets.QMessageBox()
            # msgbox.setText('已到达终点')
            # msgbox.exec_()
        else:
            reward = -10

        return (x, y), reward

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = Map(start_pos=(0,0),end_pos=(9,9))
    window.show()
    app.exec_()
