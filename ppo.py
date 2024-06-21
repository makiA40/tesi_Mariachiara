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
    PALLET_PATH = os.path.join(BASE_PATH, 'FINALE CON PALLET.xlsx')
    LOG_FILE = 'magazzino.log'
    LOG_LEVEL = logging.DEBUG

class MagazzinoEnv(gym.Env):
    def __init__(self):
        super(MagazzinoEnv, self).__init__()

        # Configurazione del logging
        logging.basicConfig(filename=Config.LOG_FILE, level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Caricamento dei dati
        self.distanze, self.pallet_data = self._load_data()

        # Normalizzazione dei dati
        self.scaler = StandardScaler()
        self.distanze = self.scaler.fit_transform(self.distanze)

        # Definizione dello spazio delle azioni e delle osservazioni
        self.action_space = spaces.Discrete(8)  # 6 movimenti + deposito pallet + rimozione pallet
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, 5, 26), dtype=np.float32)

        # Stato iniziale
        self.reset()

        # Variabili per tracciare le distanze percorse
        self.total_distance_per_episode = []
        self.current_episode_distance = 0

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

    def _get_initial_state(self) -> np.ndarray:
        return np.zeros((2, 5, 26), dtype=np.float32)

    def _get_initial_sale_dates(self) -> Dict[Tuple[int, int, int], Optional[datetime]]:
        return {(r, l, c): None for r in range(2) for l in range(5) for c in range(26)}

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

        # Resetta la distanza per il nuovo episodio
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

        # Aggiungi la distanza percorsa per questo step alla distanza totale per l'episodio
        self.current_episode_distance += np.linalg.norm(np.array(prev_position) - np.array(self.current_position))

        return self.state, reward, self.done, False, {}

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
        reward = -0.05  # Penalità minore per ogni passo

        if action == 6:  # Deposito pallet
            reward += 10
        elif action == 7:  # Rimozione pallet
            reward += 5

        if self.current_pallet_index < len(self.pallet_data):
            current_pallet = self.pallet_data.iloc[self.current_pallet_index]
            required_zone = current_pallet['ZONA']

            try:
                required_zone = int(required_zone)
            except ValueError:
                pass

            if new_position[1] + 1 == required_zone:
                reward += 10
                distance_to_closest_empty_slot = self._find_closest_empty_slot_distance(new_position)
                reward -= distance_to_closest_empty_slot * 0.1
            else:
                distance_traveled = np.sum(np.abs(np.array(prev_position) - np.array(new_position)))
                reward -= distance_traveled * 0.1

        if self.current_pallet_index >= len(self.pallet_data):
            reward += 100

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

def plot_results(results, title):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plotting mean reward
    axs[0, 0].plot(results['timesteps'], results['ep_rew_mean'])
    axs[0, 0].set_title(f'{title} - Mean Reward per Episode')
    axs[0, 0].set_xlabel('Timesteps')
    axs[0, 0].set_ylabel('Mean Reward')

    # Plotting episode length
    axs[0, 1].plot(results['timesteps'], results['ep_len_mean'])
    axs[0, 1].set_title(f'{title} - Mean Episode Length')
    axs[0, 1].set_xlabel('Timesteps')
    axs[0, 1].set_ylabel('Episode Length')

    # Plotting value loss
    axs[1, 0].plot(results['timesteps'], results['value_loss'])
    axs[1, 0].set_title(f'{title} - Value Loss')
    axs[1, 0].set_xlabel('Timesteps')
    axs[1, 0].set_ylabel('Value Loss')

    # Plotting policy gradient loss
    axs[1, 1].plot(results['timesteps'], results['policy_gradient_loss'])
    axs[1, 1].set_title(f'{title} - Policy Gradient Loss')
    axs[1, 1].set_xlabel('Timesteps')
    axs[1, 1].set_ylabel('Policy Gradient Loss')

    plt.tight_layout()
    plt.show()

    # Plotting distances
    plt.figure(figsize=(10, 5))
    plt.plot(results['timesteps'], results['distances'])
    plt.title(f'{title} - Total Distance per Episode')
    plt.xlabel('Timesteps')
    plt.ylabel('Total Distance')
    plt.show()

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
        results['timesteps'].append(timesteps)
        results['ep_rew_mean'].append(np.mean(env.total_distance_per_episode))
        results['ep_len_mean'].append(np.mean(env.total_distance_per_episode))
        results['value_loss'].append(np.mean(env.total_distance_per_episode))
        results['policy_gradient_loss'].append(np.mean(env.total_distance_per_episode))
        results['distances'].append(np.mean(env.total_distance_per_episode))
        timesteps += 8192
    return results

# Creazione dell'ambiente e addestramento del modello
env = MagazzinoEnv()
model = PPO(
    "MlpPolicy", env, verbose=1, n_steps=8192, batch_size=256, n_epochs=10, learning_rate=5e-4,
    ent_coef=0.005, clip_range=0.2, gamma=0.98, gae_lambda=0.9, vf_coef=0.5, max_grad_norm=0.5
)

# Raccogliere i risultati del modello addestrato
trained_results = collect_results(env, model, total_timesteps=500000)
plot_results(trained_results, title="Trained Model")

# Eseguire l'ambiente con azioni casuali e raccogliere i risultati
random_env = MagazzinoEnv()

random_results = {
    'timesteps': [],
    'ep_rew_mean': [],
    'ep_len_mean': [],
    'value_loss': [],
    'policy_gradient_loss': [],
    'distances': []
}
timesteps = 0
total_timesteps = 500000

while timesteps < total_timesteps:
    obs = random_env.reset()
    done = False
    while not done:
        action = random_env.action_space.sample()  # Sceglie un'azione casuale
        obs, reward, done, _, _ = random_env.step(action)
        if done:
            random_results['timesteps'].append(timesteps)
            random_results['ep_rew_mean'].append(np.mean(random_env.total_distance_per_episode))
            random_results['ep_len_mean'].append(np.mean(random_env.total_distance_per_episode))
            random_results['value_loss'].append(np.mean(random_env.total_distance_per_episode))
            random_results['policy_gradient_loss'].append(np.mean(random_env.total_distance_per_episode))
            random_results['distances'].append(np.mean(random_env.total_distance_per_episode))
            timesteps += 8192

plot_results(random_results, title="Random Actions")

# Confrontare i risultati
plt.figure(figsize=(10, 5))
plt.plot(trained_results['timesteps'], trained_results['distances'], label='Trained Model')
plt.plot(random_results['timesteps'], random_results['distances'], label='Random Actions')
plt.title('Comparison of Total Distance per Episode')
plt.xlabel('Timesteps')
plt.ylabel('Total Distance')
plt.legend()
plt.show()
