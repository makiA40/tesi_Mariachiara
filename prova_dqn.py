import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import logging
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Tuple, Optional, Dict, List

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
        self.vertical_distances = []
        self.horizontal_distances = []
        self.total_vertical_distance = 0
        self.total_horizontal_distance = 0
        self.distance_log = []

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
        self.current_episode_distance = 0
        self.vertical_distance = 0
        self.horizontal_distance = 0
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
        return distance / 1000  # Assuming the distance is in meters and needs to be converted to kilometers

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
                closest_empty_slot = self._find_closest_empty_slot_for_period(current_pallet['ZONA'], storage_date)
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
                self.logger.warning(f"Rack {rack}, Level {level}, Channel {channel} è vuoto. Non è possibile rimuovere pallet.")

    def _find_closest_empty_slot_for_period(self, zone: str, period: datetime) -> Optional[Tuple[int, int, int]]:
        empty_slots = sorted(self.empty_slots_cache, key=lambda slot: self._calculate_distance(self.current_position, slot))
        for slot in empty_slots:
            if self.sale_dates[slot] is None or self.sale_dates[slot] >= period:
                return slot
        return None

    def _check_sale_dates(self):
        current_date = datetime.now()
        for position, sale_date in self.sale_dates.items():
            if sale_date is not None and sale_date <= current_date:
                self.state[position] = 0
                self.sale_dates[position] = None
                self.empty_slots_cache.add(position)

    def _calculate_reward(self, prev_position: Tuple[int, int, int], current_position: Tuple[int, int, int], action: int) -> float:
        distance = self._calculate_distance(prev_position, current_position)
        reward = -distance
        if action in [6, 7]:
            reward += 10
        return reward

    def _check_done(self) -> bool:
        if self.current_pallet_index >= len(self.pallet_data):
            self.logger.info("Tutti i pallets sono stati processati. Episodio concluso.")
            return True
        return False

    def render(self, mode: str = 'human'):
        print(self.state)

    def export_distance_log(self):
        df = pd.DataFrame(self.distance_log)
        df.to_excel("distance_log.xlsx", index=False)

    def close(self):
        self.logger.info("Chiusura dell'ambiente.")

def plot_mean_reward(results: Dict[str, List[float]], title: str, filename: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(results['timesteps'], results['ep_rew_mean'], color='b')
    plt.title(f'{title} - Mean Reward per Episode', fontsize=14)
    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"ppo_{filename.replace('.xlsx', '')}_mean_reward.png")
    plt.show()
    plt.close()

def collect_results(env: DummyVecEnv, model: PPO, total_timesteps: int) -> Dict[str, List[float]]:
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

        monitor_env = env.envs[0]
        ep_info_buffer = monitor_env.get_episode_rewards()
        ep_len_buffer = monitor_env.get_episode_lengths()

        logger = model.logger
        value_loss_buffer = logger.name_to_value.get('train/value_loss', [])
        policy_gradient_loss_buffer = logger.name_to_value.get('train/policy_gradient_loss', [])

        mean_reward = np.mean(ep_info_buffer) if ep_info_buffer else 0
        mean_ep_len = np.mean(ep_len_buffer) if ep_len_buffer else 0
        mean_value_loss = np.mean(value_loss_buffer) if value_loss_buffer else 0
        mean_policy_gradient_loss = np.mean(policy_gradient_loss_buffer) if policy_gradient_loss_buffer else 0

        results['timesteps'].append(timesteps)
        results['ep_rew_mean'].append(mean_reward)
        results['ep_len_mean'].append(mean_ep_len)
        results['value_loss'].append(mean_value_loss)
        results['policy_gradient_loss'].append(mean_policy_gradient_loss)

        results['vertical_distances'].append(sum(monitor_env.vertical_distances))
        results['horizontal_distances'].append(sum(monitor_env.horizontal_distances))

        monitor_env.reset()
    return results

def main():
    env = DummyVecEnv([lambda: Monitor(MagazzinoEnv())])
    model = PPO('MlpPolicy', env, verbose=1)
    results = collect_results(env, model, total_timesteps=507904)

    model.save("ppo_magazzino")
    env.envs[0].export_distance_log()

    plot_mean_reward(results, "PPO", "magazzino_results.xlsx")

    # Stampa delle metriche
    print("Epoche:")
    print(results['timesteps'])
    print("Ricompensa Media per Episodio:")
    print(results['ep_rew_mean'])
    print("Lunghezza Media degli Episodi:")
    print(results['ep_len_mean'])
    print("Value Loss:")
    print(results['value_loss'])
    print("Policy Gradient Loss:")
    print(results['policy_gradient_loss'])
    print("Distanze Verticali per Episodio:")
    print(results['vertical_distances'])
    print("Distanze Orizzontali per Episodio:")
    print(results['horizontal_distances'])

if __name__ == "__main__":
    main()
