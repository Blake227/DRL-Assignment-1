# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym


class QLearningAgent():
    def __init__(self, episodes=5000, alpha=0.1, gamma=0.9, epsilon = 0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        try:
            with open("q_table.pkl", "rb") as f:
                self.q_table = pickle.load(f)
        except:
            pass

    def get_state(self, obs):
        (taxi_row, taxi_col, 
         r_row, r_col, g_row, g_col, y_row, y_col, b_row, b_col,
         obstacle_north, obstacle_south, obstacle_east, obstacle_west,
         passenger_look, destination_look) = obs
        stations = {
            'R': (r_row, r_col),
            'G': (g_row, g_col),
            'Y': (y_row, y_col),
            'B': (b_row, b_col)
        }

        passenger_at = None
        destination_at = None

        if passenger_look:
            for station, pos in stations.items():
                if (taxi_row, taxi_col) == pos:
                    passenger_at = 'taxi'
                elif (abs(taxi_row - pos[0]) + abs(taxi_col - pos[1])) == 1:
                    passenger_at = station

        if destination_look:
            for station, pos in stations.items():
                if (taxi_row, taxi_col) == pos:
                    destination_at = 'taxi'
                elif (abs(taxi_row - pos[0]) + abs(taxi_col - pos[1])) == 1:
                    destination_at = station


        state = (obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_at if passenger_at else 'none', destination_at if destination_at else 'none')
        if state not in self.q_table:
            self.q_table[state] = np.zeros(6)
        return state


    def get_action(self, obs):
        state = self.get_state(obs)
        if random.random() < self.epsilon:
            return random.randint(0, 5)
        else:
            return np.argmax(self.q_table[state])
        
    def update_q_table(self, obs, action, reward, next_obs, done):
        current_state = self.get_state(obs)
        next_state = self.get_state(next_obs)

        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (1 - done)
        td_error = td_target - self.q_table[current_state][action]
        self.q_table[current_state][action] += self.alpha * td_error

    def save_q_table(self):
        with open("q_table.pkl", "wb") as f:
            pickle.dump(self.q_table, f)


agent = QLearningAgent()


def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.


    # return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.
    return agent.get_action(obs)



