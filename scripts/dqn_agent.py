import numpy as np
import random
from collections import deque
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, model, actions, hyperparameters):
        self.model = model
        self.actions = actions
        self.learning_rate = hyperparameters["learning_rate"]
        self.epsilon = hyperparameters["exploration_factor"]
        self.epsilon_min = hyperparameters["exploration_factor_min"]
        self.epsilon_decay = hyperparameters["exploration_decay"]
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=self.optimizer, loss='mse')
        self.replay_buffer = deque(maxlen=hyperparameters["replay_buffer_size"])
        self.batch_size = hyperparameters["batch_size"]
        self.discount_factor = hyperparameters["discount_factor"]

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.actions)
        else:
            q_values = self.model.predict(state,verbose=0)
            return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.discount_factor * np.max(self.model.predict(next_state,verbose=0)[0])
            target_f = self.model.predict(state,verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def build_agent(model,actions,hyperparameters):
    agent = DQNAgent(model=model, actions=actions,hyperparameters=hyperparameters)
    return agent