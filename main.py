import random
import gym
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.exploration_rate = 1.0
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.brain = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.brain.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            self.brain.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay


class Temp:
    def __init__(self, env):
        self.env = gym.make(env)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.agent = Agent(self.state_size, self.action_size)

        self.sample_batch_size = 20
        self.episodes = 100

    def graph(self, data, title):
        plt.plot(data)
        plt.title(title)
        plt.show()

    def printRewards(self, data):
        for i in range(len(data)):
            print(str(i * 10) + "-" + str(i * 10 + 9) + ": " + str(data[i]))

    def run(self):
        average = []
        scores = []
        avg_reward = 0
        for e in range(1, self.episodes + 1):
            episode_reward = 0
            done = False
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            while not done:
                self.env.render()
                action = self.agent.act(state)
                next_state, reward, done, info = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                episode_reward = episode_reward + reward
            scores.append(episode_reward)
            avg_reward = avg_reward + episode_reward
            if e % 10 == 0 and e != 0:
                avg_reward = avg_reward / 10.0
                average.append(avg_reward)
                avg_reward = 0

            self.agent.replay(self.sample_batch_size)
        self.env.close()

        self.graph(average, 'Average of 10 episodes')
        self.graph(scores, 'Scores for each episodes')
        self.printRewards(average)





if __name__ == "__main__":
    temp = Temp('MsPacman-ram-v0')
    temp.run()
