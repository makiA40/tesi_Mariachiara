from stable_baselines3.common.vec_env import DummyVecEnv
from environment import MagazzinoEnv
from train import train_and_collect_results, plot_results
from config import load_pallet_data, load_distanze, PALLET_PATH, DISTANZE_PATH

def main():
    pallet_df = load_pallet_data(PALLET_PATH)
    distanze_normalizzate = load_distanze(DISTANZE_PATH)

    print(f"Colonne disponibili in pallet_df: {pallet_df.columns}")

    env = DummyVecEnv([lambda: MagazzinoEnv(distanze_normalizzate, pallet_df)])
    results = train_and_collect_results(env, total_timesteps=10000)
    plot_results(results)

# commento di prova

if __name__ == '__main__':
    main()
