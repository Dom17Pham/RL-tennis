# Import necessary libraries and packages
import numpy as np
import gymnasium as gym
from ale_py import ALEInterface
from ale_py.roms import Tennis
from datetime import datetime
import time
from build_model import build_model
from dqn_agent import build_agent

# Load the ROM for Atari Tennis
ale = ALEInterface()
ale.loadROM(Tennis)

# Hyperparameters
hyperparameters = {
    "learning_rate": 0.1,           # Learning Rate
    "exploration_factor": 1.0,      # Exploration Factor (initial)
    "exploration_factor_min":0.1,   # Minimum Exploration Factor
    "exploration_decay":0.995,      # Exploration Decay
    "discount_factor": 0.99,        # Discount Factor
    "training_episodes": 10,      # Number of training episodes
    "testing_episodes": 1,          # Number of testing episodes
    "max_steps": 100,               # Maximum steps per episode
    "replay_buffer_size": 10000,
    "batch_size": 32,
    "save_interval": 500
}

# Initialize the environment
env = gym.make('ALE/Tennis-v5',full_action_space=True)
env.reset() 

actions = env.action_space.n 
height, width, channels = env.observation_space.shape

model = build_model(height, width, channels, actions)

agent = build_agent(model,actions,hyperparameters)

# Training loop
performance_record = []  # List to store performance of each episode
best_average_reward = -np.inf  
average_reward = 0
best_model_filename = None
start_time = time.time()

for episode in range(hyperparameters["training_episodes"]):
    state_tuple = env.reset()
    observation = state_tuple[0]  # Extract the observation from the tuple
    state = np.expand_dims(observation, axis=0)  # Add batch dimension
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        observations = env.step(action)
        next_state = observations[0]
        reward = observations[1]
        done = observations[2]
        next_state = np.expand_dims(next_state, axis=0)  # Add batch dimension
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break
    agent.replay()
    performance_record.append(total_reward)

    # Save the model weights every save_interval episodes
    if (episode + 1) % hyperparameters["save_interval"] == 0:
        model_filename = f"agent_model_episode_{episode + 1}.keras"
        model.save(model_filename)
    
    # Track the best performing model
    average_reward = total_reward
    if average_reward > best_average_reward:
        best_average_reward = average_reward
        model.save("../models/best_model.keras")

    print(f"--- Episode {episode + 1} finished ---")
    
end_time = time.time()
training_duration = end_time - start_time
hours, rem = divmod(training_duration, 3600)
minutes, seconds = divmod(rem, 60)

print(f"Training Completed in {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")

env.close()