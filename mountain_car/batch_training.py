import multiprocessing
import sys
import os
import pickle

from tqdm import tqdm

from mountain_car.training import run_training
import mountain_car.plotting as plots
from models.vae_dense_obs import DenseObservation_VAE
from models.biased_model import BiasedModelBellman

num_simulations = 15
experiment_dir = './experiments/batch_run/'
save_models = False
num_episodes = 200
time_compression = 6
observation_noise_std = None  # (0.05, 0.05)
agent_parameters = dict(
            vae=DenseObservation_VAE(
                observation_dim=2,
                latent_dim=2),
            biased_model=BiasedModelBellman(observation_dim=2, iterate_train=10, discount_factor=0.995),
            # biased_model=distr.Normal(torch.tensor([0.5, 0.0]), torch.tensor([1.0, 1.0])),
            policy_dim=1,
            time_step_size=time_compression,
            planning_horizon=5,
            n_policy_samples=700,
            policy_iterations=3,
            n_policy_candidates=70,
            action_window=2,
            # disable_kl_extrinsic=True,  # Uncomment for ablation study
            # disable_kl_intrinsic=True   # Uncomment for ablation study
        )


def run_training_process(training_id):
    return run_training(
        agent_parameters=agent_parameters,
        time_compression=time_compression,
        episode_callbacks=[],
        save_dirpath=experiment_dir if save_models else None,
        episodes=num_episodes,
        observation_noise_std=observation_noise_std,
        model_id=training_id,
        verbose=False)


if __name__ == "__main__":
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    num_cpu_cores = multiprocessing.cpu_count()
    num_processes = min(num_simulations, num_cpu_cores)
    print(f'\nRunning {num_simulations} simulations in {num_processes} parallel processes...')
    with multiprocessing.Pool(processes=num_processes) as pool:
        all_results = []
        for new_result in tqdm(pool.imap_unordered(run_training_process, range(num_simulations)), total=num_simulations, file=sys.stdout):
            all_results.append(new_result)
            with open(os.path.join(experiment_dir, 'results.pickle'), 'wb') as f:
                pickle.dump(all_results, f)
            plots.plot_training_history(all_results, save_path=os.path.join(experiment_dir, 'run_stats.pdf'), show=False)
