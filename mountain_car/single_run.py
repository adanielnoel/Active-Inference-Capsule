import os
import sys
import pickle

from tqdm import tqdm
import numpy as np
import gym
import torch
import torch.distributions as distr

from models.vae_dense_obs import DenseObservation_VAE
from models.active_inference_capsule import ActiveInferenceCapsule
from models.biased_model import BiasedModelBellman
from utils.timeline import Timeline
from utils.value_map import ValueMap
from utils.model_saving import load_capsule_parameters
import mountain_car.plotting as plots

# Note: scroll to bottom for running as script


def run_training(agent_parameters,              # Agent parameters
                 time_compression,              # Simulation parameter
                 episodes,                      # Simulation parameter
                 observation_noise_std=None,    # Simulation parameter
                 model_id=None,                 # Simulation parameter
                 episode_callbacks=(),          # Job setting
                 save_dirpath=None,             # Job setting
                 save_all_episodes=False,       # Job setting
                 load_existing=False,           # Job setting
                 load_vae=True,                 # Job setting
                 load_transition_model=True,    # Job setting
                 load_biased_model=True,        # Job setting
                 train_parameters=True,         # Job setting
                 verbose=True,                  # Job setting
                 display_simulation=False):     # Job setting

    env = gym.make('MountainCarContinuous-v0').env
    observations_mapper = ValueMap(in_min=torch.tensor((-1.2, -0.07)), in_max=torch.tensor((0.6, 0.07)),
                                   out_min=torch.tensor((-1.0, -1.0)), out_max=torch.tensor((1.0, 1.0)))
    aif_agent = ActiveInferenceCapsule(**agent_parameters)

    if save_dirpath is not None:
        # Create directory if it doesn't exist
        if not os.path.exists(save_dirpath):
            os.makedirs(save_dirpath)
        model_save_filepath = os.path.join(save_dirpath, f'model{model_id or ""}.pt')

        # Save agent and simulation configuration
        with open(os.path.join(save_dirpath, f'settings{model_id or ""}.pk'), 'wb') as f:
            pickle.dump(dict(agent_parameters=agent_parameters,
                             time_compression=time_compression,
                             episodes=episodes,
                             observation_noise_std=observation_noise_std,
                             model_id=model_id), f)

        # Load previous model
        if os.path.exists(model_save_filepath) and load_existing:
            load_capsule_parameters(aif_agent, model_save_filepath, load_vae, load_transition_model, load_biased_model)
            if verbose:
                loaded_models = []
                loaded_models += ['vae'] if load_vae else []
                loaded_models += ['transition_model'] if load_transition_model else []
                loaded_models += ['biased_model'] if load_biased_model and not isinstance(aif_agent.biased_model, distr.Normal) else []
                print(f"\nLoaded <{', '.join(loaded_models)}> from previous save at <{model_save_filepath}>")
    else:
        model_save_filepath = None

    if not train_parameters:
        aif_agent.eval()
    max_episode_steps = 1000
    training_history = Timeline()
    for episode in range(episodes):
        env.reset()
        aif_agent.reset_states()
        state = observations_mapper(torch.from_numpy(env.state).float())
        action = aif_agent.step(0, state)
        total_reward = 0
        episode_history = Timeline()
        episode_history.log(0, 'true_observations', state)
        iterator = tqdm(range(max_episode_steps), file=sys.stdout, disable=not verbose)
        iterator.set_description(f'Trial {episode + 1}/{episodes}')
        i = 0
        for i in iterator:
            if display_simulation:
                env.render()
            t = i + 1
            observation, reward, done, _ = env.step(action)
            observation = observations_mapper(torch.from_numpy(observation).float())
            episode_history.log(t, 'true_observations', observation)
            episode_history.log(t - 1, 'true_actions', action)
            if i % time_compression == 0:
                obs_noise = observation if observation_noise_std is None else observation + np.random.normal(loc=0.0, scale=observation_noise_std)
                action = aif_agent.step(t, obs_noise, action)
                action = np.clip(action, env.min_action, env.max_action)
            total_reward += reward

            if done:
                for callback in episode_callbacks:
                    callback(**dict(agent=aif_agent,
                                    env=env,
                                    episode_reward=reward,
                                    episode_history=episode_history,
                                    observations_mapper=observations_mapper))
                aif_agent.learn_biased_model()  # Train biased model with the successful trajectory
                iterator.set_postfix_str(f'reward={total_reward:.2f}')
                break
            elif i == max_episode_steps - 1:
                iterator.set_postfix_str(f'reward={total_reward:.2f}')
        if save_dirpath is not None:
            torch.save(aif_agent.state_dict(), model_save_filepath if not save_all_episodes else os.path.join(save_dirpath, f'model{model_id or ""}_{episode}.pt'))
        training_history.log(episode, 'steps_per_episode', i)
        training_history.log(episode, 'rewards', total_reward)

    env.close()
    return training_history


if __name__ == '__main__':
    plot = False
    experiment_dir = './experiments/single_run/'
    _time_compression = 6
    _model_id = None

    from time import time

    t0 = time()
    res = run_training(
        agent_parameters=dict(
            vae=DenseObservation_VAE(
                observation_dim=2,
                latent_dim=2),
            biased_model=BiasedModelBellman(observation_dim=2, iterate_train=10, discount_factor=0.995),
            # biased_model=distr.Normal(torch.tensor([0.5, 0.0]), torch.tensor([1.0, 1.0])),
            policy_dim=1,
            time_step_size=_time_compression,
            planning_horizon=10,
            n_policy_samples=700,
            policy_iterations=3,
            n_policy_candidates=70,
            action_window=2,
            # disable_kl_extrinsic=True,  # Uncomment for ablation study
            # disable_kl_intrinsic=True   # Uncomment for ablation study
        ),
        time_compression=_time_compression,
        episodes=10,
        observation_noise_std=None,
        model_id=_model_id,
        episode_callbacks=[plots.show_phase_portrait, plots.show_prediction_vs_outcome] if plot else [],
        save_dirpath=experiment_dir,
        save_all_episodes=False,
        load_existing=False,
        train_parameters=True,
        verbose=True,
        display_simulation=False
    )
    print(f'Finished {len(res.times)} episodes in {time() - t0:.1f} seconds')
    with open(os.path.join(experiment_dir, f'results{_model_id or ""}.pickle'), 'wb') as f:
        pickle.dump(res, f)
    plots.plot_training_history(res, save_path=os.path.join(experiment_dir, f'run_stats{_model_id or ""}.pdf'), show=False)
