# train_dqn_cartpole_pytorch.py
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

# ----- Hyperparameters -----
EPISODES = 1000
GAMMA = 0.95
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 2000

# ----- DQN Model -----
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)  # ✅ fix: hidden layer -> action size

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ----- Agent -----
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)

            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state)).item()

            target_f = self.model(state).detach().clone()
            target_f[0][action] = target

            output = self.model(state)
            loss = self.criterion(output, target_f)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ----- Training Loop -----
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

scores = []

for e in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, truncated, _ = env.step(action)
        reward = reward if not done else -10
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        agent.replay(BATCH_SIZE)

    scores.append(total_reward)
    print(f"Episode: {e+1}/{EPISODES}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")

# ----- Save Model -----
torch.save(agent.model.state_dict(), "dqn_cartpole_week5.pth")
print("✅ Model saved as dqn_cartpole_week5.pth")

# ----- Plot Training -----
plt.plot(scores)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("DQN Training on CartPole-v1 (PyTorch)")
plt.savefig("training_cartpole_week5.png")
plt.show()
