import sys
import random
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

# 定义常量
CELL_SIZE = 30  # 单元格的大小
GRID_SIZE = 10  # 格子的大小
PADDING = 20  # 内边距
MARGIN = 50  # 边距

# 定义颜色
COLOR_START = QColor(0, 255, 0)  # 起点颜色
COLOR_END = QColor(255, 0, 0)  # 终点颜色
COLOR_PATH = QColor(255, 255, 0)  # 路径颜色
COLOR_BLOCKED = QColor(0, 0, 0)  # 障碍物颜色
COLOR_BACKGROUND = QColor(255, 255, 255)  # 背景颜色

class GridWorld(QWidget):
    def __init__(self, grid_size, start_pos, end_pos, path_list):
        super().__init__()
        
        # 设置窗口固定大小
        self.setFixedSize((grid_size + 2) * CELL_SIZE + PADDING * 2, (grid_size + 2) * CELL_SIZE + PADDING * 2)
        
        # 初始化内部变量
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.path_list = path_list
        self.epsilon = 1
        self.alpha = 0.5
        self.gamma = 0.5
        self.q_table = {}
        self.current_pos = start_pos

        # 初始化 Q 表格
        for i in range(grid_size):
            for j in range(grid_size):
                state = (i, j)
                self.q_table[state] = {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0}

        # 设置定时器，用于不断更新界面
        self.timer = QTimer(self)
        self.timer.timeout.connect()
        self.timer.start(100)

    # 给定一个状态，返回对应的行动（使用 epsilon-greedy 策略）
    def get_action(self, state):
        if random.uniform(0.0, 1.0) < self.epsilon:
            return random.choice(["up", "down", "left", "right"])
        else:
            q_values = self.q_table[state]
            max_q = max(q_values.values())
            actions = [a for a, q in q_values.items() if q == max_q]
            return random.choice(actions)

    # 根据当前状态和行动，返回下一个状态和奖励
    def step(self, state, action):
        row, col = state
        if action == "up":
            row = max(row - 1, 0)
        elif action == "down":
            row = min(row + 1, self.grid_size - 1)
        elif action == "left":
            col = max(col - 1, 0)
        elif action == "right":
            col = min(col + 1, self.grid_size - 1)

        next_state = (row, col)

        if next_state in self.path_list:
            reward = 0.0
        else:
            reward = -1.0

        return next_state, reward

    # 更新 Q 表格
    def update_q_table(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value

    # 渲染界面
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制背景
        painter.fillRect(self.rect(), COLOR_BACKGROUND)

        # 绘制路径
        for i in range(len(self.path_list)):
            x, y = self.path_list[i]
            painter.fillRect(PADDING + (x + 1) * CELL_SIZE, PADDING + (y + 1) * CELL_SIZE, CELL_SIZE, CELL_SIZE, COLOR_PATH)

        # 绘制起点和终点
        painter.fillRect(PADDING + (self.start_pos[0] + 1) * CELL_SIZE, PADDING + (self.start_pos[1] + 1) * CELL_SIZE, CELL_SIZE, CELL_SIZE, COLOR_START)
        painter.fillRect(PADDING + (self.end_pos[0] + 1) * CELL_SIZE, PADDING + (self.end_pos[1] + 1) * CELL_SIZE, CELL_SIZE, CELL_SIZE, COLOR_END)

        # 绘制小车
        painter.fillRect(PADDING + (self.current_pos[0] + 1) * CELL_SIZE, PADDING + (self.current_pos[1] + 1) * CELL_SIZE, CELL_SIZE, CELL_SIZE, QColor(128, 128, 255))
        


    # 处理按键事件（用于手动控制小车）
    def keyPressEvent(self, event):
        row, col = self.current_pos
        if event.key() == Qt.Key_Up:
            row = max(row - 1, 0)
        elif event.key() == Qt.Key_Down:
            row = min(row + 1, self.grid_size - 1)
        elif event.key() == Qt.Key_Left:
            col = max(col - 1, 0)
        elif event.key() == Qt.Key_Right:
            col = min(col + 1, self.grid_size - 1)

        next_pos = (row, col)
        if next_pos in self.path_list:
            self.current_pos = next_pos
            self.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 定义路线图（黄色代表路线）
    path_list = [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3),(2,3),(2,4), (4, 3), (5, 3), (5, 2), (5, 1), (6, 1), (7, 1), (8, 1)]
    
    # 定义起点和终点
    start_pos = (0, 0)
    end_pos = (9, 1)
    
    # 创建界面对象
    world = GridWorld(GRID_SIZE, start_pos, end_pos, path_list)
    
    
    # 迭代更新 Q 表格
    for i in range(100):
        state = start_pos
        while state != end_pos:
            action = world.get_action(state)
            next_state, reward = world.step(state, action)
            world.update_q_table(state, action, reward, next_state)
            state = next_state

        # 重置小车的位置，以便重新开始寻路
        world.current_pos = start_pos    
        

    
    # 显示界面
    world.show()
    sys.exit(app.exec_())
