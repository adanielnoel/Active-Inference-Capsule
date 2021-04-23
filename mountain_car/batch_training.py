import multiprocessing
import sys
import os
import pickle

from tqdm import tqdm
import torch
import torch.distributions as distr

from mountain_car.training import run_training
from models.vae_dense_obs import DenseObservation_VAE
from models.biased_model import BiasedModelBellman

num_simulations = 15
max_cpus = None
num_episodes = 150
save_models = False
experiment_dir = './experiments/batch_run/'
name = 'results_bellman60_noise_position'
_learn_biased_model = True
_include_cart_velocity = False
_observation_noise_std = 0.08
_time_compression = 6
_planning_horizon = 10  # Multiply with _time_compression to get in simulation steps

if _learn_biased_model:
    biased_model = BiasedModelBellman(observation_dim=2 if _include_cart_velocity else 1, iterate_train=10, discount_factor=0.995)
else:
    biased_model = distr.Normal(torch.tensor([0.9, 0.0]) if _include_cart_velocity else 0.9, 1.0)

agent_parameters = dict(
    vae=DenseObservation_VAE(
        observation_dim=2 if _include_cart_velocity else 1,
        latent_dim=2 if _include_cart_velocity else 1,
        observation_noise_std=_observation_noise_std),
    biased_model=biased_model,
    policy_dim=1,
    time_step_size=_time_compression,
    planning_horizon=_planning_horizon,
    n_policy_samples=700,
    policy_iterations=4,
    n_policy_candidates=70,
    action_window=2
)


def run_training_process(training_id):
    return run_training(
        agent_parameters=agent_parameters,
        time_compression=_time_compression,
        episode_callbacks=[],
        save_dirpath=experiment_dir if save_models else None,
        episodes=num_episodes,
        observation_noise_std=_observation_noise_std,
        include_cart_velocity=_include_cart_velocity,
        model_id=training_id,
        verbose=False)


if __name__ == "__main__":
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    num_cpu_cores = max_cpus or multiprocessing.cpu_count()
    num_processes = min(num_simulations, num_cpu_cores)
    print(f'\nRunning {num_simulations} simulations of "{name}" in {num_processes} parallel processes...')
    from time import time
    t0 = time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        all_results = []
        for new_result in tqdm(pool.imap_unordered(run_training_process, range(num_simulations)), total=num_simulations, file=sys.stdout):
            all_results.append(new_result)
            with open(os.path.join(experiment_dir, f'{name}.pickle'), 'wb') as f:
                pickle.dump(all_results, f)
    t1 = time() - t0
    print(f'Finished {num_episodes * num_simulations} episodes in {t1/60:.1f} minutes ({t1 / (num_episodes * num_simulations):.2f}s/episode)')
    # plots.plot_training_history(all_results, save_path=os.path.join(experiment_dir, 'run_stats.pdf'), show=False)
