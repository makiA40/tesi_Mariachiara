import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import logging
from typing import Tuple, Optional, Dict
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
        self.scaler = StandardScaler()
        self.distanze = self.scaler.fit_transform(self.distanze)
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, 5, 26), dtype=np.float32)
        self.reset()
        self.total_distance_per_episode = []
        self.current_episode_distance = 0

    def setup_logger(self):
        logging.basicConfig(filename=Config.LOG_FILE, level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            distanze = pd.read_excel(Config.DISTANZE_PATH)
            pallet_data = pd.read_excel(Config.PALLET_PATH)
            self.logger.info("Dati caricati con successo.")
            return distanze, pallet_data
        except ImportError as e:
            self.logger.error(f"Libreria mancante: {e}. Assicurati di avere installato 'openpyxl'.")
            raise
        except Exception as e:
            self.logger.error(f"Errore nel caricamento dei file: {e}")
            raise

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.seed(seed)
        self.state = self._get_initial_state()
        self.sale_dates = self._get_initial_sale_dates()
        self.done = False
        self.current_position = (0, 0, 0)
        self.current_pallet_index = 0
        self.empty_slots_cache = {(r, l, c) for r in range(2) for l in range(5) for c in range(26) if self.state[r, l, c] == 0}
        self.logger.info("Ambiente resettato.")
        if hasattr(self, 'current_episode_distance'):
            self.total_distance_per_episode.append(self.current_episode_distance)
        self.current_episode_distance = 0
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
        self.logger.info(f"Step: {self.current_pallet_index}, Action: {action}, Position: {self.current_position}, Reward: {reward}")
        self.current_episode_distance += np.linalg.norm(np.array(prev_position) - np.array(self.current_position))
        return self.state, reward, self.done, False, {}

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
                closest_empty_slot = self._find_closest_empty_slot_same_company(company, storage_date)
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
                self.current_position = (r, l, c)
                self.logger.info(f"Stored {pallets_to_store} pallets at position {(r, l, c)}. Remaining pallets to store: {num_pallets}")
                if self.state[r, l, c] == max_capacity:
                    self.empty_slots_cache.remove((r, l, c))
            self.current_pallet_index += 1
        else:
            if self.state[rack, level, channel] > 0:
                self.state[rack, level, channel] -= 1
                if self.state[rack, level, channel] == 0:
                    self.sale_dates[(rack, level, channel)] = None
                    self.empty_slots_cache.add((rack, level, channel))
                self.logger.info(f"Rimosso 1 pallet dalla posizione {(rack, level, channel)}")
            else:
                self.logger.warning(f"Rack {rack}, Level {level}, Channel {channel} è vuoto. Non è possibile rimuovere pallet.")

    def _check_sale_dates(self):
        current_date = datetime.now()
        for position, sale_date in self.sale_dates.items():
            if sale_date is not None and sale_date <= current_date:
                rack, level, channel = position
                self.state[rack, level, channel] = 0
                self.sale_dates[position] = None
                self.empty_slots_cache.add((rack, level, channel))
                self.logger.info(f"Slot {position} è stato liberato sulla base della data di vendita {sale_date}.")

    def _calculate_reward(self, prev_position: Tuple[int, int, int], new_position: Tuple[int, int, int], action: int) -> float:
        reward = -0.1
        if action == 6:
            reward += 20
        elif action == 7:
            reward += 10
        if self.current_pallet_index < len(self.pallet_data):
            current_pallet = self.pallet_data.iloc[self.current_pallet_index]
            required_zone = current_pallet['ZONA']
            try:
                required_zone = int(required_zone)
            except ValueError:
                pass
            if new_position[1] + 1 == required_zone:
                reward += 20
                distance_to_closest_empty_slot = self._find_closest_empty_slot_distance(new_position)
                reward -= distance_to_closest_empty_slot * 0.2
            else:
                distance_traveled = np.sum(np.abs(np.array(prev_position) - np.array(new_position)))
                reward -= distance_traveled * 0.2
        if self.current_pallet_index >= len(self.pallet_data):
            reward += 200
        return reward

    def _check_done(self) -> bool:
        return self.current_pallet_index >= len(self.pallet_data)

    def render(self, mode='human') -> None:
        print(f"Current state: {self.state}")
        print(f"Current position: {self.current_position}")
        print(f"Current pallet index: {self.current_pallet_index}")

    def _find_closest_empty_slot_same_company(self, company: str, storage_date: datetime) -> Optional[Tuple[int, int, int]]:
        for r, l, c in self.empty_slots_cache:
            if self.sale_dates[(r, l, c)] == storage_date:
                return r, l, c
        return self._find_closest_empty_slot()

    def _find_closest_empty_slot_distance(self, position: Tuple[int, int, int]) -> float:
        distances = [
            self.distanze[position[0]*5*26 + position[1]*26 + position[2], empty_slot[0]*5*26 + empty_slot[1]*26 + empty_slot[2]]
            for empty_slot in self.empty_slots_cache
        ]
        return min(distances) if distances else float('inf')

    def _find_closest_empty_slot(self) -> Optional[Tuple[int, int, int]]:
        if self.empty_slots_cache:
            return min(self.empty_slots_cache, key=lambda slot: self.distanze[
                slot[0]*5*26 + slot[1]*26 + slot[2], self.current_position[0]*5*26 + self.current_position[1]*26 + self.current_position[2]
            ])
        return None

def plot_results(results, title, filename):
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    metrics = ['ep_rew_mean', 'ep_len_mean', 'value_loss', 'policy_gradient_loss', 'distances']
    colors = ['b', 'g', 'r', 'c', 'm']
    titles = [
        'Mean Reward per Episode', 'Mean Episode Length',
        'Value Loss', 'Policy Gradient Loss',
        'Total Distance per Episode'
    ]
    for i, (metric, color, sub_title) in enumerate(zip(metrics, colors, titles)):
        axs[i // 2, i % 2].plot(results['timesteps'], results[metric], color=color)
        axs[i // 2, i % 2].set_title(f'{title} - {sub_title}')
        axs[i // 2, i % 2].set_xlabel('Timesteps')
        axs[i // 2, i % 2].set_ylabel(sub_title.split()[-1])
        axs[i // 2, i % 2].grid(True)
    axs[2, 1].plot(results['timesteps'], results['ep_rew_mean'], label='Mean Reward', color='b')
    axs[2, 1].plot(results['timesteps'], results['ep_len_mean'], label='Episode Length', color='g')
    axs[2, 1].plot(results['timesteps'], results['distances'], label='Total Distance', color='m')
    axs[2, 1].set_title(f'{title} - Reward, Length, and Distance Over Time')
    axs[2, 1].set_xlabel('Timesteps')
    axs[2, 1].set_ylabel('Value')
    axs[2, 1].legend()
    axs[2, 1].grid(True)
    plt.suptitle(f'Results from {filename}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{filename.replace('.xlsx', '')}_results.png")
    plt.close()

def collect_results(env, model, total_timesteps):
    results = {
        'timesteps': [],
        'ep_rew_mean': [],
        'ep_len_mean': [],
        'value_loss': [],
        'policy_gradient_loss': [],
        'distances': []
    }
    timesteps = 0
    while timesteps < total_timesteps:
        model.learn(total_timesteps=8192, reset_num_timesteps=False)
        timesteps += 8192
        results['timesteps'].append(timesteps)
        results['ep_rew_mean'].append(np.mean(env.total_distance_per_episode))
        results['ep_len_mean'].append(np.mean(env.total_distance_per_episode))
        results['value_loss'].append(np.mean(env.total_distance_per_episode))
        results['policy_gradient_loss'].append(np.mean(env.total_distance_per_episode))
        results['distances'].append(np.mean(env.total_distance_per_episode))
    return results

env = MagazzinoEnv()
model = PPO(
    "MlpPolicy", env, verbose=1, n_steps=8192, batch_size=256, n_epochs=10, learning_rate=1e-3,
    ent_coef=0.01, clip_range=0.2, gamma=0.98, gae_lambda=0.9, vf_coef=0.5, max_grad_norm=0.5
)
trained_results = collect_results(env, model, total_timesteps=500000)
plot_results(trained_results, title="Trained Model", filename="magazzino.xlsx")
