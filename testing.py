import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- Definisi model (sama persis dengan training) ---
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- Load Environment ---
env = gym.make("CartPole-v1", render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# --- Load model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(state_size, action_size).to(device)
model.load_state_dict(torch.load("dqn_cartpole_week5.pth", map_location=device))
model.eval()

scores = []
EPISODES = 5

for e in range(EPISODES):
    state, _ = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    done = False
    score = 0

    while not done:
        with torch.no_grad():
            q_values = model(state)
        action = torch.argmax(q_values, dim=1).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        score += reward

        state = torch.FloatTensor(next_state).unsqueeze(0).to(device)

    scores.append(score)
    print(f"Test Episode: {e+1}/{EPISODES}, Score: {score}")

env.close()

# --- Simpan grafik hasil test ---
plt.plot(scores, marker='o')
plt.title("Test Performance on CartPole-v1")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.ylim(0, 500)
plt.grid()
plt.savefig("test_cartpole_week5.png")
plt.show()
