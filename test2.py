import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ReplayBuffer(object):
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, state, action, next_state, reward, done):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = (state, action, next_state, reward, done)
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, next_states, rewards, dones = [], [], [], [], []
        for i in ind:
            s, a, s_, r, d = self.storage[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            next_states.append(np.array(s_, copy=False))
            rewards.append(np.array(r, copy=False))
            dones.append(np.array(d, copy=False))
        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards).reshape(-1, 1), np.array(dones).reshape(-1, 1)

class Agent(object):
    def __init__(self, input_dim, output_dim, env_name, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, replay_buffer_size=1000000, batch_size=64):
        self.env = gym.make(env_name) 
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)
        self.network = DQN(input_dim, output_dim)
        self.target_network = DQN(input_dim, output_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_value = self.network(state)
            action = q_value.max(1)[1].item()
        return action

    def train(self, num_episodes=5000):
        scores = []
        for i_episode in range(num_episodes):
            state = self.env.reset()
            done = False
            score = 0.0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.add(state, action, next_state, reward, done)
                state = next_state
                score += reward

                if len(self.replay_buffer.storage) >= self.batch_size:
                    states, actions, next_states, rewards, dones = self.replay_buffer.sample(self.batch_size)
                    states = torch.FloatTensor(states).to(self.device)
                    actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
                    next_states = torch.FloatTensor(next_states).to(self.device)
                    rewards = torch.FloatTensor(rewards).to(self.device)
                    dones = torch.FloatTensor(dones).to(self.device)

                    q_values = self.network(states).gather(1, actions).squeeze(1)
                    next_q_values = self.target_network(next_states).max(1)[0]
                    targets = rewards + (1 - dones) * self.gamma * next_q_values

                    loss = F.mse_loss(q_values, targets.detach())

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    self.update_target_model()

            scores.append(score)
            print('Episode: {} - Score: {} - Epsilon: {:.3f}'.format(i_episode, score, self.epsilon))

            self.epsilon *= self.epsilon_decay
        
        return scores

    def update_target_model(self):
        self.target_network.load_state_dict(self.network.state_dict())
    
    def save_model(self, path):
        torch.save(self.network.state_dict(), path)
        
    def load_model(self, path):
        self.network.load_state_dict(torch.load(path))


# 创建一个CartPole-v0的环境，CartPole是一个经典的强化学习任务
env_name = 'CartPole-v1'
env = gym.make(env_name)

# 检查状态空间和动作空间的维度
print(f'State space dimension: {env.observation_space.shape}')
print(f'Action space dimension: {env.action_space.n}')

# 设置agent和训练参数
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
agent = Agent(input_dim, output_dim, env_name)
scores = agent.train(num_episodes=100)
