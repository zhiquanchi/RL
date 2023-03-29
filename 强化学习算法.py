import numpy as np

# 定义状态空间、动作空间和奖励函数
num_states = 100 # 分成100个状态
num_actions = 5 # 包括前进、后退、左转、右转、停止
gamma = 0.9 # 折扣因子
alpha = 0.5 # 学习率

# 初始化 Q 表
Q = np.zeros((num_states, num_actions))

# 获取当前状态和可选动作
def get_state_and_actions(pos):
    state = int(pos * num_states / 100) # 通过将环境分为 num_states 个格子来离散化状态
    actions = [0, 1, 2, 3, 4] # 前进、后退、左转、右转、停止，编号为 0~4
    if state == 0: # 起点状态只能执行前进操作
        actions = [0]
    if state == num_states - 1: # 终点状态只能执行停止操作
        actions = [4]
    return state, actions

# 开始训练
for episode in range(100):
    pos = 0 # 初始位置在起点
    done = False # 是否结束
    while not done:
        state, actions = get_state_and_actions(pos)
        action = np.argmax(Q[state, actions]) # 使用贪心策略选择最优动作
        next_pos = pos + (action - 2) # 执行动作，计算下一个位置
        if next_pos < 0: # 超出左端点处理
            next_pos = 0
        elif next_pos > 100: # 超出右端点处理
            next_pos = 100
        next_state, _ = get_state_and_actions(next_pos)
        reward = -abs(next_pos - 45) # 奖励函数，距离终点越近奖励越高
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action]) # 更新 Q 表
        pos = next_pos # 更新当前位置
        if pos == 100: # 到达终点
            done = True
        # print('第',episode,'次训练')


# 测试
pos = 0 # 初始位置在起点
done = False # 是否结束
while not done:
    state, actions = get_state_and_actions(pos)
    action = np.argmax(Q[state, actions]) # 使用贪心策略选择最优动作
    next_pos = pos + (action - 2) # 执行动作，计算下一个位置
    if next_pos < 0: # 超出左端点处理
        next_pos = 0
    elif next_pos > 100: # 超出右端点处理
        next_pos = 100
    pos = next_pos # 更新当前位置
    if pos == 100: # 到达终点
        done = True
    print("当前位置：{}，采取行动：{}，到达位置：{}".format(pos, action, next_pos))
