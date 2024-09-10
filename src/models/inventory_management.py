# src/model_q_learning.py

import numpy as np
import pandas as pd

class InventoryManagementModel:
    def __init__(self, config):
        self.config = config
        self.states = range(self.config["inventory"]["max_inventory"])
        self.actions = range(self.config["inventory"]["max_order"])
        self.q_table = np.zeros((len(self.states), len(self.actions)))

    def train_model(self, demand_data):
        """Train Q-learning model."""
        alpha = self.config["q_learning"]["alpha"]
        gamma = self.config["q_learning"]["gamma"]
        epsilon = self.config["q_learning"]["epsilon"]

        for episode in range(self.config["q_learning"]["episodes"]):
            state = np.random.choice(self.states)
            for step in range(len(demand_data)):
                action = self.epsilon_greedy_policy(state, epsilon)
                reward, next_state = self.simulate_environment(state, action, demand_data.iloc[step])
                self.q_table[state, action] = (1 - alpha) * self.q_table[state, action] + alpha * (reward + gamma * np.max(self.q_table[next_state]))
                state = next_state

        print("Q-learning training completed.")
        return self.q_table

    def epsilon_greedy_policy(self, state, epsilon):
        """Choose an action based on epsilon-greedy policy."""
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(self.q_table[state])

    def simulate_environment(self, state, action, demand):
        """Simulate the inventory environment and compute reward."""
        next_state = max(0, state - demand + action)
        reward = -abs(demand - action)  # Simplified reward calculation
        return reward, next_state

    def run(self, demand_data):
        """Run inventory management model."""
        self.train_model(demand_data)

if __name__ == "__main__":
    config = ConfigParser()
    config.read("config/config.yaml")
    im_model = InventoryManagementModel(config)
    demand_data = pd.read_csv(config["data"]["demand_data_path"])
    im_model.run(demand_data)
