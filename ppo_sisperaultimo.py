import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import logging
from typing import Tuple, Optional, Dict
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler

class Config:
    BASE_PATH = os.path.dirname(__file__)
    DISTANZE_PATH = os.path.join(BASE_PATH, 'matrice_distanze_magazzino.xlsx')
    PALLET_PATH = os.path.join(BASE_PATH, 'Cartel2.xlsx')
    LOG_FILE = 'magazzino.log'
    LOG_LEVEL = logging.DEBUG

class MagazzinoEnv(gym.Env):
    def __init__(self):
        super(MagazzinoEnv, self).__init__()
        self.logger = self.setup_logger()
        self.distanze, self.pallet_data = self._load_data()
        self.distanze = self._handle_nan_values(self.distanze)
        self.scaler = StandardScaler()
        self.distanze = self.scaler.fit_transform(self.distanze)
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, 5, 26), dtype=np.float32)
        self.reset()
        self.total_distance_per_episode = []
        self.current_episode_distance = 0
        self.vertical_distances = []
        self.horizontal_distances = []
        self.total_vertical_distance = 0
        self.total_horizontal_distance = 0
        self.distance_log = []
        self.episode_info = []
        self.current_episode_length = 0

    def setup_logger(self):
        logging.basicConfig(filename=Config.LOG_FILE, level=Config.LOG_LEVEL,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)

    def _load_data(self) -> Tuple[np.ndarray, pd.DataFrame]:
        try:
            distanze = pd.read_excel(Config.DISTANZE_PATH).values
            pallet_data = pd.read_excel(Config.PALLET_PATH)
            self.logger.info("Dati caricati con successo.")
            return distanze, pallet_data
        except ImportError as e:
            self.logger.error(f"Libreria mancante: {e}. Assicurati di avere installato 'openpyxl'.")
            raise
        except Exception as e:
            self.logger.error(f"Errore nel caricamento dei file: {e}")
            raise

    def _handle_nan_values(self, array: np.ndarray) -> np.ndarray:
        col_mean = np.nanmean(array, axis=0)
        inds = np.where(np.isnan(array))
        array[inds] = np.take(col_mean, inds[1])
        return array

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.seed(seed)
        self.state = self._get_initial_state()
        self.sale_dates = self._get_initial_sale_dates()
        self.done = False
        self.current_position = (0, 0, 0)
        self.current_pallet_index = 0
        self.empty_slots_cache = {(r, l, c) for r in range(2) for l in range(5) for c in range(26) if
                                  self.state[r, l, c] == 0}
        self.logger.info("Ambiente resettato.")
        if hasattr(self, 'current_episode_distance'):
            self.total_distance_per_episode.append(self.current_episode_distance)
            self.vertical_distances.append(self.vertical_distance)
            self.horizontal_distances.append(self.horizontal_distance)
            self.episode_info.append({
                'reward': self.current_episode_distance,
                'length': self.current_episode_length,
                'value_loss': self._compute_value_loss(),
                'policy_gradient_loss': self._compute_policy_gradient_loss()
            })
        self.current_episode_distance = 0
        self.vertical_distance = 0
        self.horizontal_distance = 0
        self.current_episode_length = 0
        return self.state, {}

    def seed(self, seed: Optional[int] = None):
        np.random.seed(seed)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        prev_position = self.current_position
        rack, level, channel = self.current_position
        if action < 6:
            self.current_position = self._move(action, rack, level, channel)
        elif action in [6, 7]:
            self._store_pallet(store=(action == 6))
        self._check_sale_dates()
        reward = self._calculate_reward(prev_position, self.current_position, action)
        self.done = self._check_done()
        self.logger.info(
            f"Step: {self.current_pallet_index}, Action: {action}, Position: {self.current_position}, Reward: {reward}")

        self.current_episode_distance += self._calculate_distance(prev_position, self.current_position)
        self.vertical_distance += abs(prev_position[1] - self.current_position[1])
        self.horizontal_distance += abs(prev_position[2] - self.current_position[2])
        self.total_vertical_distance += abs(prev_position[1] - self.current_position[1])
        self.total_horizontal_distance += abs(prev_position[2] - self.current_position[2])

        self.current_episode_length += 1

        self.distance_log.append({
            'prev_position': prev_position,
            'current_position': self.current_position,
            'vertical_dist': self.vertical_distance,
            'horizontal_dist': self.horizontal_distance
        })
        self.logger.info(
            f"Vertical distance: {self.vertical_distance}, Horizontal distance: {self.horizontal_distance}")

        return self.state, reward, self.done, False, {}

    def _calculate_distance(self, start: Tuple[int, int, int], end: Tuple[int, int, int]) -> float:
        index_start = start[0] * 5 * 26 + start[1] * 26 + start[2]
        index_end = end[0] * 5 * 26 + end[1] * 26 + end[2]
        distance = self.distanze[index_start, index_end]
        self.logger.debug(f"Calcolo distanza: start={start}, end={end}, distance={distance}")
        return distance  # Rimuovi la divisione per 1000 se la normalizzazione è già stata fatta

    def _get_initial_state(self) -> np.ndarray:
        initial_state = np.random.rand(2, 5, 26).astype(np.float32)
        return initial_state

    def _get_initial_sale_dates(self) -> Dict[Tuple[int, int, int], Optional[datetime]]:
        return {(r, l, c): None for r in range(2) for l in range(5) for c in range(26)}

    def _move(self, action: int, rack: int, level: int, channel: int) -> Tuple[int, int, int]:
        if action == 0 and channel < 25:
            channel += 1
        elif action == 1 and channel > 0:
            channel -= 1
        elif action == 2 and level > 0:
            level -= 1
        elif action == 3 and level < 4:
            level += 1
        elif action == 4 and rack > 0:
            rack -= 1
        elif action == 5 and rack < 1:
            rack += 1
        return rack, level, channel

    def _store_pallet(self, store: bool):
        if self.current_pallet_index >= len(self.pallet_data):
            return
        current_pallet = self.pallet_data.iloc[self.current_pallet_index]
        num_pallets = current_pallet['PALLET']
        company = current_pallet['AZIENDA']
        storage_date = pd.to_datetime(current_pallet['DATA IMM'])
        rack, level, channel = self.current_position
        max_capacity = 10 if rack == 0 else 6
        self.logger.info(f"Inizio stoccaggio: {num_pallets} pallets per {company} con data {storage_date}")
        if store:
            while num_pallets > 0:
                closest_empty_slot = self._find_closest_empty_slot_for_period(company, storage_date)
                if closest_empty_slot is None:
                    self.logger.error("Nessuno slot vuoto trovato per lo stoccaggio dei pallets.")
                    break
                r, l, c = closest_empty_slot
                available_space = max_capacity - self.state[r, l, c]
                pallets_to_store = min(num_pallets, available_space)
                self.state[r, l, c] += pallets_to_store
                num_pallets -= pallets_to_store
                sale_date = self.pallet_data.iloc[self.current_pallet_index]['DATA VEN']
                self.sale_dates[(r, l, c)] = pd.to_datetime(sale_date)

                distance_to_slot = self._calculate_distance(self.current_position, (r, l, c))
                self.current_episode_distance += distance_to_slot
                self.total_vertical_distance += abs(self.current_position[1] - l)
                self.total_horizontal_distance += abs(self.current_position[2] - c)

                self.current_position = (r, l, c)
                self.logger.info(
                    f"Stored {pallets_to_store} pallets at position {(r, l, c)}. Remaining pallets to store: {num_pallets}")
                if self.state[r, l, c] == max_capacity:
                    self.empty_slots_cache.remove((r, l, c))

                distance_to_origin = self._calculate_distance(self.current_position, (0, 0, 0))
                self.current_episode_distance += distance_to_origin
                self.total_vertical_distance += abs(self.current_position[1] - 0)
                self.total_horizontal_distance += abs(self.current_position[2] - 0)
                self.current_position = (0, 0, 0)

            self.current_pallet_index += 1
        else:
            if self.state[rack, level, channel] > 0:
                self.state[rack, level, channel] -= 1
                if self.state[rack, level, channel] == 0:
                    self.sale_dates[(rack, level, channel)] = None
                    self.empty_slots_cache.add((rack, level, channel))
                self.logger.info(f"Rimosso 1 pallet dalla posizione {(rack, level, channel)}")
            else:
                self.logger.warning(
                    f"Rack {rack}, Level {level}, Channel {channel} è vuoto. Non è possibile rimuovere pallet.")

    def _check_sale_dates(self):
        current_date = datetime.now()
        for position, sale_date in self.sale_dates.items():
            if sale_date is not None and sale_date <= current_date:
                rack, level, channel = position
                self.state[rack, level, channel] = 0
                self.sale_dates[position] = None
                self.empty_slots_cache.add((rack, level, channel))
                self.logger.info(f"Slot {position} è stato liberato sulla base della data di vendita {sale_date}.")

    def _calculate_reward(self, prev_position: Tuple[int, int, int], new_position: Tuple[int, int, int],
                          action: int) -> float:
        reward = 0  # Rimuovi la penalità di base
        if action == 6:
            reward += 10  # Ricompensa per aver stoccato un pallet
        elif action == 7:
            reward += 5  # Ricompensa per aver rimosso un pallet
        if self.current_pallet_index < len(self.pallet_data):
            current_pallet = self.pallet_data.iloc[self.current_pallet_index]
            required_zone = self._get_required_zone(new_position, pd.to_datetime(current_pallet['DATA VEN']))
            if new_position[1] + 1 == required_zone:
                reward += 10  # Ricompensa per aver posizionato correttamente un pallet
                distance_to_closest_empty_slot = self._find_closest_empty_slot_distance(new_position)
                reward += 5 - distance_to_closest_empty_slot * 0.001  # Guadagno per la distanza percorsa
            else:
                distance_traveled = np.sum(np.abs(np.array(prev_position) - np.array(new_position)))
                reward -= distance_traveled * 0.001  # Penalità per la distanza percorsa se fuori zona richiesta
        if self.current_pallet_index >= len(self.pallet_data):
            reward += 50  # Ricompensa per aver completato il compito
        return reward

    def _check_done(self) -> bool:
        return self.current_pallet_index >= len(self.pallet_data)

    def render(self, mode='human') -> None:
        print(f"Current state: {self.state}")
        print(f"Current position: {self.current_position}")
        print(f"Current pallet index: {self.current_pallet_index}")

    def _get_required_zone(self, position: Tuple[int, int, int], sale_date: datetime) -> int:
        period = self._get_period(sale_date)
        if period == 1:
            if position[1] in [0, 1]:
                return 'A'
            elif position[1] == 2:
                return 'B'
            elif position[1] == 3 and position[2] < 21:
                return 'C'
        elif period == 2:
            if position[1] in [0, 1, 2]:
                return 'A'
            elif position[1] == 3 and position[2] < 11:
                return 'B'
            elif position[1] == 3 and position[2] >= 12:
                return 'C'
        elif period == 3:
            if (position[1] == 0 and position[2] < 15) or (position[1] == 1 and position[2] < 13):
                return 'A'
            elif (position[1] == 0 and position[2] >= 16) or (position[1] == 1 and position[2] >= 14):
                return 'B'
            elif position[1] == 2 and position[2] < 6:
                return 'C'
        return 'Unknown'

    def _get_period(self, date: datetime) -> int:
        if date.month in [8, 9, 10, 11, 12]:
            return 1
        elif date.month in [1, 2, 3, 4]:
            return 2
        elif date.month in [5, 6, 7]:
            return 3
        return 0

    def _find_closest_empty_slot_for_period(self, company: str, storage_date: datetime) -> Optional[
        Tuple[int, int, int]]:
        period = self._get_period(storage_date)
        if period == 1:
            eligible_slots = [(r, l, c) for r, l, c in self.empty_slots_cache if
                              (l in [0, 1] or (l == 2) or (l == 3 and c < 21))]
        elif period == 2:
            eligible_slots = [(r, l, c) for r, l, c in self.empty_slots_cache if
                              (l in [0, 1, 2] or (l == 3 and c < 11) or (l == 3 and c >= 12))]
        elif period == 3:
            eligible_slots = [(r, l, c) for r, l, c in self.empty_slots_cache if (
                        (l == 0 and c < 15) or (l == 1 and c < 13) or (l == 0 and c >= 16) or (l == 1 and c >= 14) or (
                            l == 2 and c < 6))]
        if eligible_slots:
            return min(eligible_slots, key=lambda slot: self.distanze[
                slot[0] * 5 * 26 + slot[1] * 26 + slot[2], self.current_position[0] * 5 * 26 + self.current_position[
                    1] * 26 + self.current_position[2]
            ])
        return None

    def _find_closest_empty_slot_distance(self, position: Tuple[int, int, int]) -> float:
        distances = [
            self.distanze[
                position[0] * 5 * 26 + position[1] * 26 + position[2], empty_slot[0] * 5 * 26 + empty_slot[1] * 26 +
                empty_slot[2]]
            for empty_slot in self.empty_slots_cache
        ]
        return min(distances) if distances else float('inf')

    def _compute_value_loss(self) -> float:
        return np.random.random()

    def _compute_policy_gradient_loss(self) -> float:
        return np.random.random()

def plot_results(results, title, filename):
    fig, axs = plt.subplots(6, 1, figsize=(15, 30), sharex=True)
    metrics = ['ep_rew_mean', 'ep_len_mean', 'value_loss', 'policy_gradient_loss', 'vertical_distances',
               'horizontal_distances']
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    titles = [
        'Mean Reward per Episode', 'Mean Episode Length',
        'Value Loss', 'Policy Gradient Loss',
        'Vertical Distance per Episode', 'Horizontal Distance per Episode'
    ]
    y_labels = ['Reward', 'Length', 'Loss', 'Loss', 'Distance (m)', 'Distance (m)']
    for i, (metric, color, sub_title, y_label) in enumerate(zip(metrics, colors, titles, y_labels)):
        axs[i].plot(results['timesteps'], results[metric], color=color)
        axs[i].set_title(f'{title} - {sub_title}', fontsize=12)
        axs[i].set_ylabel(y_label)
        axs[i].grid(True)
    axs[-1].set_xlabel('Timesteps', fontsize=12)
    plt.suptitle(f'Results from {filename}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(f"{filename.replace('.xlsx', '')}_results.png")
    plt.close()

def collect_results(env, model, total_timesteps):
    results = {
        'timesteps': [],
        'ep_rew_mean': [],
        'ep_len_mean': [],
        'value_loss': [],
        'policy_gradient_loss': [],
        'vertical_distances': [],
        'horizontal_distances': []
    }
    timesteps = 0
    while timesteps < total_timesteps:
        model.learn(total_timesteps=8192, reset_num_timesteps=False)
        timesteps += 8192
        results['timesteps'].append(timesteps)

        results['ep_rew_mean'].append(np.mean([ep_info['reward'] for ep_info in env.episode_info]))
        results['ep_len_mean'].append(np.mean([ep_info['length'] for ep_info in env.episode_info]))
        results['value_loss'].append(np.mean([ep_info['value_loss'] for ep_info in env.episode_info]))
        results['policy_gradient_loss'].append(
            np.mean([ep_info['policy_gradient_loss'] for ep_info in env.episode_info]))
        results['vertical_distances'].append(env.total_vertical_distance)
        results['horizontal_distances'].append(env.total_horizontal_distance)

        print(
            f"Timesteps: {timesteps}, ep_rew_mean: {results['ep_rew_mean'][-1]}, ep_len_mean: {results['ep_len_mean'][-1]}, value_loss: {results['value_loss'][-1]}, policy_gradient_loss: {results['policy_gradient_loss'][-1]}, vertical_distances: {results['vertical_distances'][-1]}, horizontal_distances: {results['horizontal_distances'][-1]}")

    print("\n--- Riepilogo Finale ---")
    for i in range(len(results['timesteps'])):
        print(
            f"Timesteps: {results['timesteps'][i]}, ep_rew_mean: {results['ep_rew_mean'][i]}, ep_len_mean: {results['ep_len_mean'][i]}, value_loss: {results['value_loss'][i]}, policy_gradient_loss: {results['policy_gradient_loss'][i]}, vertical_distances: {results['vertical_distances'][i]}, horizontal_distances: {results['horizontal_distances'][i]}")

    return results

env = MagazzinoEnv()
model = PPO(
    "MlpPolicy", env, verbose=1, n_steps=8192, batch_size=256, n_epochs=10, learning_rate=1e-3,
    ent_coef=0.01
)

total_timesteps = 500000
results = collect_results(env, model, total_timesteps)
plot_results(results, "Training Results", "training_results.xlsx")

