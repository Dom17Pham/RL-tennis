# Import necessary libraries and packages
import numpy as np
import gymnasium as gym
from ale_py import ALEInterface
from ale_py.roms import Tennis
from tensorflow.keras.models import load_model
from build_model import build_model
from dqn_agent import build_agent

# Load the ROM for Atari Tennis
ale = ALEInterface()
ale.loadROM(Tennis)

# Hyperparameters
hyperparameters = {
    "learning_rate": 0.1,           # Learning Rate
    "exploration_factor": 1.0,      # Exploration Factor 
    "exploration_factor_min":0.1,   # Minimum Exploration Factor
    "exploration_decay":0.995,      # Exploration Decay
    "discount_factor": 0.99,        # Discount Factor
    "training_episodes": 1,         # Number of training episodes
    "testing_episodes": 1,          # Number of testing episdoes
    "max_steps": 100,               # Maximum steps per episode
    "replay_buffer_size": 10000,
    "batch_size": 32,
    "save_interval": 100
}

# Initialize the environment
env = gym.make('ALE/Tennis-v5',full_action_space=True)
env.reset() 

actions = env.action_space.n 
height, width, channels = env.observation_space.shape

model = build_model(height, width, channels, actions)

# Load the saved model
saved_model_path = "../models/best_model.keras"  
loaded_model = load_model(saved_model_path)

# Create a new agent with the loaded model
testing_agent = build_agent(model,actions,hyperparameters)

env = gym.make('ALE/Tennis-v5',full_action_space=True, render_mode="human")
env.reset() 

# Testing loop
for episode in range(hyperparameters["testing_episodes"]):
    state_tuple = env.reset()
    observation = state_tuple[0]  # Extract the observation from the tuple
    state = np.expand_dims(observation, axis=0)  # Add batch dimension
    total_reward = 0
    done = False
    while not done:
        env.render()
        action = testing_agent.act(state)
        observations = env.step(action)
        next_state = observations[0]
        reward = observations[1]
        done = observations[2]
        next_state = np.expand_dims(next_state, axis=0)  # Add batch dimension
        state = next_state
        total_reward += reward
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
env.close()