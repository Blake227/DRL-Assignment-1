import gym
import numpy as np
from student_agent import agent
import pickle
from simple_custom_taxi_env import SimpleTaxiEnv
from tqdm import tqdm

def train_agent(episodes=10000):
    env = SimpleTaxiEnv(grid_size=5, fuel_limit=5000)

    for episode in tqdm(range(episodes)):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.update_q_table(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
            
        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}")
            agent.save_q_table()  # Periodically save the Q-table
    
    # Save the final Q-table
    agent.save_q_table()
    print("Training completed.")

if __name__ == "__main__":
    train_agent()