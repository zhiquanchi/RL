import numpy as np
import matplotlib.pyplot as plt
import random

states = list(range(16))
actions = ['up', 'down', 'left', 'right']

rewards = np.array([
    [-1, -1, -1, -1],
    [-1, -1, -1,  0],
    [-1, -1, -1,  0],
    [-1,  0,  0,  1]
])

transitions = np.zeros((16, 4, 16))
for i in range(16):
    for j in range(4):
        if i == 0 or i == 15:
            transitions[i][j][i] = 1
        else:
            if i == 0: #top
                transitions[i][j][i] = 1
            if j == 0:  # up
                if i < 4:
                    transitions[i][j][i] = 1
                else:
                    transitions[i][j][i-4] = 1
            elif j == 1:  # down
                if i > 11:
                    transitions[i][j][i] = 1
                else:
                    transitions[i][j][i+4] = 1
            elif j == 2:  # left
                if i % 4 == 0:
                    transitions[i][j][i] = 1
                else:
                    transitions[i][j][i-1] = 1
            elif j == 3:  # right
                if i % 4 == 3:
                    transitions[i][j][i] = 1
                else:
                    transitions[i][j][i+1] = 1

q_table = np.zeros((16, 4))

alpha = 0.5
gamma = 0.9
epsilon = 0.1

num_episodes = 1000
for episode in range(num_episodes):
    state = random.randint(0, 15)
    while state != 0 and state != 15:
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            valid_actions = [a for a in actions if transitions[state][actions.index(a)].sum() > 0]
            action = random.choice(valid_actions) if valid_actions else random.choice(actions)
        # 检查state和action是否合法
        if state < 0 or state > 15:
            raise ValueError("Invalid state: {}".format(state))
        if action not in actions:
            raise ValueError("Invalid action: {}".format(action))
        next_state = np.random.choice(states, p=transitions[state][actions.index(action)])
        reward = rewards[state][actions.index(action)]
        # 检查reward是否合法
        if reward is None:
            raise ValueError("Invalid reward for state {} and action {}: {}".format(state, action, reward))
        q_table[state][actions.index(action)] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][actions.index(action)])
        state = next_state


state = 0
optimal_path = [state]
while state != 15:
    action = actions[int(np.argmax(q_table[state]))]
    next_state = np.random.choice(states, p=transitions[state][actions.index(action)])
    state = next_state
    optimal_path.append(state)

fig, ax = plt.subplots()
maze = np.zeros((4, 4))
maze[0, 0] = 1
maze[3, 3] = 1
ax.matshow(maze, cmap=plt.cm.Blues)
for i in range(4):
    for j in range(4):
        if maze[i, j] == 0:
            s = str(i*4+j)
            ax.text(j, i, s, ha='center', va='center', color='w')
for i in range(len(optimal_path)):
    if i == 0:
        pass
    else:
        ax.plot((optimal_path[i-1] % 4, optimal_path[i] % 4), (optimal_path[i-1] // 4, optimal_path[i] // 4), c='r')
plt.show()
