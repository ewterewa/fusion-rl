import numpy as np
import matplotlib.pyplot as plt
from model.plasma_model import PlasmaModel
from model.replay_buffer import ReplayBuffer

class SimpleDQN:
    def __init__(self, state_size=8, action_size=5):
        self.state_size = state_size
        self.action_size = action_size
        self.weights = np.random.randn(action_size, state_size) * 0.1
        self.lr = 0.01
        self.gamma = 0.95
        self.epsilon = 0.3
        self.buffer = ReplayBuffer(capacity=2000)
        self.episode_reward = 0
    
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.weights @ state
        return np.argmax(q_values)
    
    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self.episode_reward += reward
    
    def replay(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        for i in range(batch_size):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]
            done = dones[i]
            if done:
                target = reward
            else:
                next_q = self.weights @ next_state
                target = reward + self.gamma * np.max(next_q)
            current = self.weights[action] @ state
            loss = target - current
            self.weights[action] += self.lr * loss * state
        self.epsilon = max(0.01, self.epsilon * 0.999)

def train(episodes=200):
    env = PlasmaModel()
    agent = SimpleDQN()
    rewards = []
    for ep in range(episodes):
        state = env.reset(random_init=True)
        agent.episode_reward = 0
        for step in range(100):
            action = agent.act(state)
            next_state = env.step(action)
            reward = env.calculate_reward()
            done = env.is_disrupted()
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            if done:
                break
        rewards.append(agent.episode_reward)
        if (ep+1) % 20 == 0:
            avg = np.mean(rewards[-20:])
            print(f"Эпизод {ep+1}, средняя награда: {avg:.2f}")
    return rewards

if __name__ == "__main__":
    rewards = train(episodes=200)
    plt.plot(rewards)
    plt.xlabel("Эпизод")
    plt.ylabel("Награда")
    plt.title("Обучение DQN")
    plt.savefig("rewards.png")
    print("Обучение завершено, график сохранен")
