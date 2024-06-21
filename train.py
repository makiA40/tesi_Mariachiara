import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.env = env
        self.episode_rewards = []
        self.episode_lengths = []
        self.value_losses = []
        self.policy_gradient_losses = []
        self.vertical_distances = []
        self.horizontal_distances = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        episode_rewards = self.env.get_attr('episode_rewards')[0]
        episode_lengths = len(episode_rewards)
        vertical_distances = self.env.get_attr('vertical_distances')[0]
        horizontal_distances = self.env.get_attr('horizontal_distances')[0]

        self.episode_rewards.append(np.sum(episode_rewards))
        self.episode_lengths.append(episode_lengths)
        self.vertical_distances.append(vertical_distances)
        self.horizontal_distances.append(horizontal_distances)

        ep_rew_mean = np.mean(self.episode_rewards)
        ep_len_mean = np.mean(self.episode_lengths)
        vertical_dist_mean = np.mean(self.vertical_distances)
        horizontal_dist_mean = np.mean(self.horizontal_distances)

        value_loss = self.locals.get('value_loss', np.nan)
        policy_gradient_loss = self.locals.get('policy_loss', np.nan)

        self.value_losses.append(value_loss)
        self.policy_gradient_losses.append(policy_gradient_loss)

        print(f"Timesteps: {self.num_timesteps}, ep_rew_mean: {ep_rew_mean}, ep_len_mean: {ep_len_mean}, "
              f"value_loss: {value_loss}, policy_gradient_loss: {policy_gradient_loss}, "
              f"vertical_distances: {vertical_dist_mean}, horizontal_distances: {horizontal_dist_mean}")


def train_and_collect_results(env, total_timesteps):
    model = PPO("MlpPolicy", env, verbose=1)
    custom_callback = CustomCallback(env)
    model.learn(total_timesteps=total_timesteps, callback=custom_callback)

    results = {
        "timesteps": np.arange(0, total_timesteps, model.n_steps),
        "ep_rew_mean": custom_callback.episode_rewards,
        "ep_len_mean": custom_callback.episode_lengths,
        "value_losses": custom_callback.value_losses,
        "policy_gradient_losses": custom_callback.policy_gradient_losses,
        "vertical_distances": custom_callback.vertical_distances,
        "horizontal_distances": custom_callback.horizontal_distances,
    }
    return results

def plot_results(results):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.plot(results["timesteps"], results["ep_rew_mean"])
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward')

    plt.subplot(2, 3, 2)
    plt.plot(results["timesteps"], results["ep_len_mean"])
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Episode Length')
    plt.title('Mean Episode Length')

    plt.subplot(2, 3, 3)
    plt.plot(results["timesteps"], results["value_losses"])
    plt.xlabel('Timesteps')
    plt.ylabel('Value Loss')
    plt.title('Value Loss')

    plt.subplot(2, 3, 4)
    plt.plot(results["timesteps"], results["policy_gradient_losses"])
    plt.xlabel('Timesteps')
    plt.ylabel('Policy Gradient Loss')
    plt.title('Policy Gradient Loss')

    plt.subplot(2, 3, 5)
    plt.plot(results["timesteps"], results["vertical_distances"])
    plt.xlabel('Timesteps')
    plt.ylabel('Vertical Distances')
    plt.title('Vertical Distances')

    plt.subplot(2, 3, 6)
    plt.plot(results["timesteps"], results["horizontal_distances"])
    plt.xlabel('Timesteps')
    plt.ylabel('Horizontal Distances')
    plt.title('Horizontal Distances')

    plt.tight_layout()
    plt.show()
