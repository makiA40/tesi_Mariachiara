import gym
from gym import spaces
import numpy as np
import pandas as pd
from datetime import datetime

class MagazzinoEnv(gym.Env):
    def __init__(self, distanze, pallet_data):
        super(MagazzinoEnv, self).__init__()

        self.distanze = distanze.values  # Convertiamo il DataFrame in numpy array
        self.pallet_data = pallet_data

        if len(self.distanze.shape) == 2:
            self.n_racks = 2
            self.n_levels = 5
            self.n_channels = 26
        else:
            raise ValueError("La matrice delle distanze deve avere 2 dimensioni")

        self.action_space = spaces.Discrete(self.n_racks * self.n_levels * self.n_channels)
        self.observation_space = spaces.Box(low=0, high=10, shape=(self.n_racks, self.n_levels, self.n_channels), dtype=np.int32)

        self.magazzino = np.zeros((self.n_racks, self.n_levels, self.n_channels), dtype=np.int32)
        self.current_position = (0, 0, 0)
        self.current_pallet_index = 0

        self.capacity_per_channel = np.array([
            [10] * self.n_channels if rack == 0 else [6] * self.n_channels
            for rack in range(self.n_racks)
            for level in range(self.n_levels)
        ]).reshape(self.n_racks, self.n_levels, self.n_channels)

        self.episode_rewards = []
        self.vertical_distances = 0
        self.horizontal_distances = 0

    def reset(self):
        self.magazzino = np.zeros((self.n_racks, self.n_levels, self.n_channels), dtype=np.int32)
        self.current_position = (0, 0, 0)
        self.current_pallet_index = 0
        self.episode_rewards = []
        self.vertical_distances = 0
        self.horizontal_distances = 0
        return self.magazzino

    def step(self, action):
        rack = action // (self.n_levels * self.n_channels)
        level = (action % (self.n_levels * self.n_channels)) // self.n_channels
        channel = action % self.n_channels

        prev_position = self.current_position
        self.current_position = (rack, level, channel)

        reward = self._calculate_reward(prev_position, self.current_position, action)
        self.episode_rewards.append(reward)

        prev_index = prev_position[0] * self.n_levels * self.n_channels + prev_position[1] * self.n_channels + prev_position[2]
        current_index = self.current_position[0] * self.n_levels * self.n_channels + self.current_position[1] * self.n_channels + self.current_position[2]
        distance_cost = self.distanze[prev_index, current_index]

        self.vertical_distances += abs(prev_position[1] - level)
        self.horizontal_distances += abs(prev_position[2] - channel)

        reward -= distance_cost

        done = self.current_pallet_index >= len(self.pallet_data) - 1
        self.current_pallet_index += 1

        info = {'distance_cost': distance_cost}

        return self.magazzino, reward, done, info

    def _calculate_reward(self, prev_position, current_position, action):
        current_pallet = self.pallet_data.iloc[self.current_pallet_index]
        data_ven = current_pallet['DATA VEN']
        num_pallet = current_pallet['PALLET']

        if isinstance(data_ven, pd.Timestamp):
            data_ven = data_ven.strftime("%Y-%m-%d")

        month = datetime.strptime(data_ven, "%Y-%m-%d").month

        if 8 <= month <= 12:
            periodo = 1
        elif 1 <= month <= 4:
            periodo = 2
        elif 5 <= month <= 7:
            periodo = 3
        else:
            periodo = None

        zona = current_pallet['ZONA']
        rack, level, channel = current_position

        reward = 0

        if periodo == 1:
            if (zona == 'A' and level in [0, 1]) or \
                    (zona == 'B' and level == 2) or \
                    (zona == 'C' and level == 3 and channel <= 20):
                reward += 20  # Ricompensa aumentata per le azioni corrette
            else:
                reward -= 1  # Penalità ridotta per le azioni sbagliate
        elif periodo == 2:
            if (zona == 'A' and level in [0, 1, 2]) or \
                    (zona == 'B' and level == 3 and channel <= 10) or \
                    (zona == 'C' and level == 3 and 11 <= channel <= 25):
                reward += 20
            else:
                reward -= 1
        elif periodo == 3:
            if (zona == 'A' and ((level == 0 and channel <= 14) or (level == 1 and channel <= 12))) or \
                    (zona == 'B' and ((level == 0 and 15 <= channel <= 25) or (level == 1 and 13 <= channel <= 25))) or \
                    (zona == 'C' and level == 2 and channel <= 5):
                reward += 20
            else:
                reward -= 1

        reward -= 0.1  # Penalità ridotta per ogni passo per incentivare l'efficienza

        if self.magazzino[rack][level][channel] + num_pallet > self.capacity_per_channel[rack][level][channel]:
            reward -= 5  # Penalità ridotta per sovraccarico
        else:
            self.magazzino[rack][level][channel] += num_pallet

        self.current_position = (0, 0, 0)

        # Aggiungi un controllo finale per assicurarti che il reward sia un valore valido
        if np.isnan(reward) or np.isinf(reward):
            reward = -10

        return reward

    def render(self, mode='human'):
        pass

    def find_best_slot(self, num_pallet, periodo, zona):
        best_slot = None
        min_distance = float('inf')
        for rack in range(self.n_racks):
            for level in range(self.n_levels):
                for channel in range(self.n_channels):
                    if self.magazzino[rack][level][channel] + num_pallet <= self.capacity_per_channel[rack][level][channel]:
                        if periodo == 1 and not (
                            (zona == 'A' and level in [0, 1]) or
                            (zona == 'B' and level == 2) or
                            (zona == 'C' and level == 3 and channel <= 20)):
                            continue
                        elif periodo == 2 and not (
                            (zona == 'A' and level in [0, 1, 2]) or
                            (zona == 'B' and level == 3 and channel <= 10) or
                            (zona == 'C' and level == 3 and 11 <= channel <= 25)):
                            continue
                        elif periodo == 3 and not (
                            (zona == 'A' and ((level == 0 and channel <= 14) or (level == 1 and channel <= 12))) or
                            (zona == 'B' and ((level == 0 and 15 <= channel <= 25) or (level == 1 and 13 <= channel <= 25))) or
                            (zona == 'C' and level == 2 and channel <= 5)):
                            continue
                        current_position = (rack, level, channel)
                        distance_cost = self.calculate_distance(self.current_position, current_position)
                        if distance_cost < min_distance:
                            min_distance = distance_cost
                            best_slot = current_position
        return best_slot

    def calculate_distance(self, pos1, pos2):
        index1 = pos1[0] * self.n_levels * self.n_channels + pos1[1] * self.n_channels + pos1[2]
        index2 = pos2[0] * self.n_levels * self.n_channels + pos2[1] * self.n_channels + pos2[2]
        return self.distanze[index1, index2]

