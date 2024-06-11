import numpy as np
import random
from collections import deque
from tensorflow.keras.optimizers import Adam

class PrioritizedReplayBuffer:
    def __init__(self, size, alpha=0.6):
        self.size = size
        self.alpha = alpha
        self.buffer = deque(maxlen=size)
        self.priorities = deque(maxlen=size)
    
    def add(self, experience, error):
        priority = (error + 1e-5) ** self.alpha
        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        scaled_priorities = np.array(self.priorities) ** beta

        # Avoid NaN values
        if np.any(np.isnan(scaled_priorities)):
            scaled_priorities = np.nan_to_num(scaled_priorities, nan=0.0, posinf=0.0, neginf=0.0)
        
        sample_probs = scaled_priorities / sum(scaled_priorities)
        
        # Avoid NaN values in sample_probs
        if np.any(np.isnan(sample_probs)):
            sample_probs = np.nan_to_num(sample_probs, nan=1.0/len(sample_probs))

        indices = np.random.choice(len(self.buffer), batch_size, p=sample_probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * sample_probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            priority = (error + 1e-5) ** self.alpha
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)
    
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
        self.replay_buffer = PrioritizedReplayBuffer(hyperparameters["replay_buffer_size"])
        self.batch_size = hyperparameters["batch_size"]
        self.discount_factor = hyperparameters["discount_factor"]
        self.priority_exponent = hyperparameters["priority_exponent"]
        self.importance_sampling = hyperparameters["importance_sampling"]

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.actions)
        else:
            q_values = self.model.predict(state,verbose=0)
            return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        q_values = self.model.predict(state, verbose=0)
        next_q_values = self.model.predict(next_state, verbose=0)
        target = reward
        if not done:
            target = reward + self.discount_factor * np.max(next_q_values[0])
        error = abs(target - q_values[0][action])
        self.replay_buffer.add((state, action, reward, next_state, done), error)

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        minibatch, indices, weights = self.replay_buffer.sample(self.batch_size, self.importance_sampling)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.vstack(states)
        next_states = np.vstack(next_states)
        q_values = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)

        targets = q_values.copy()
        errors = np.zeros(self.batch_size)
        
        for i in range(self.batch_size):
            target = rewards[i]
            if not dones[i]:
                target = rewards[i] + self.discount_factor * np.max(next_q_values[i])
            errors[i] = abs(target - q_values[i][actions[i]])
            targets[i][actions[i]] = target

        self.replay_buffer.update_priorities(indices, errors)
        self.model.fit(states, targets, sample_weight=weights, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def build_agent(model,actions,hyperparameters):
    agent = DQNAgent(model=model, actions=actions,hyperparameters=hyperparameters)
    return agent