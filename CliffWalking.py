from pprint import pprint
class CliffWalkingEnv:
    """ Cliff Walking环境"""
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol # 定义环境的宽
        self.nrow = nrow # 定义环境的高
        self.P = self.createP() # 转移矩阵P[state][action] = [(p, next_state, reward, done)]，包含下一个状态和奖励

    def createP(self):
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)] # 初始化
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]] # 4 种动作, 0:上, 1:下, 2:左, 3:右。原点(0,0)定义在左上角
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    if i == self.nrow - 1 and j > 0:  # 位置在悬崖或者终点，因为无法继续交互，任何动作奖励都为0
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)]
                        continue
                    # 其他位置
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    if next_y == self.nrow - 1 and next_x > 0: # 下一个位置在悬崖或者终点
                        done = True
                        if next_x != self.ncol - 1: # 下一个位置在悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P
    
c = CliffWalkingEnv()
for i in range(1):
    pprint(c.createP())