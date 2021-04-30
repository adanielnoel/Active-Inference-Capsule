import os
import sys
import pickle

import gym
import torch
import torch.distributions as distr
import numpy as np
import tqdm

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
                 observation_noise_std=None,    # Simulation parameter
                 include_cart_velocity=True,    # Simulation parameter
                 model_id=None,                 # Simulation parameter - use to manage multiple independent models in the same folder
                 episodes=1,                    # Job setting
                 episode_callbacks=(),          # Job setting
                 frame_callbacks=(),            # Job setting
                 save_dirpath=None,             # Job setting
                 model_load_filepath=None,      # Job setting
                 save_all_episodes=False,       # Job setting
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
                             observation_noise_std=observation_noise_std,
                             include_cart_velocity=include_cart_velocity,
                             model_id=model_id), f)
    else:
        model_save_filepath = None

    # Load previous model
    if model_load_filepath is not None:
        load_capsule_parameters(aif_agent, model_load_filepath, load_vae, load_transition_model, load_biased_model)
        if verbose:
            loaded_models = []
            loaded_models += ['vae'] if load_vae else []
            loaded_models += ['transition_model'] if load_transition_model else []
            loaded_models += ['biased_model'] if load_biased_model and not isinstance(aif_agent.biased_model, distr.Normal) else []
            print(f"\nLoaded <{', '.join(loaded_models)}> from previous save at <{model_load_filepath}>")

    if save_dirpath is not None and train_parameters:
        torch.save(aif_agent.state_dict(), model_save_filepath if not save_all_episodes else os.path.join(save_dirpath, f'model{model_id or ""}_000.pt'))

    if not train_parameters:
        aif_agent.eval()
    training_history = Timeline()
    max_episode_steps = 1000
    for episode in range(episodes):
        env.reset()
        aif_agent.reset_states()
        observations_mapper = observations_mapper if observations_mapper is not None else lambda x: x
        state = observations_mapper(torch.from_numpy(env.state).float())
        state_noisy = state if observation_noise_std is None else state + torch.normal(0.0, torch.tensor(observation_noise_std))
        action = aif_agent.step(0, state_noisy if include_cart_velocity else state_noisy[[0]])
        total_reward = 0
        episode_history = Timeline()
        episode_history.log(0, 'true_observations', state)
        episode_history.log(0, 'noisy_observations', state_noisy)
        iterator = tqdm.tqdm(range(max_episode_steps), file=sys.stdout, disable=not verbose)
        iterator.set_description(f'Running episode {episode}/{episodes}')
        for i in iterator:
            if display_simulation:
                frame = env.render(mode='rgb_array')
            else:
                frame = None
            t = i + 1
            observation, reward, done, _ = env.step(action)
            observation = observations_mapper(torch.from_numpy(observation).float())
            episode_history.log(t, 'true_observations', observation)
            episode_history.log(t - 1, 'true_actions', action)
            obs_noise = observation if observation_noise_std is None else observation + torch.normal(0.0, torch.tensor(observation_noise_std))
            episode_history.log(t, 'noisy_observations', obs_noise)
            if i % time_compression == 0:
                action = aif_agent.step(t, obs_noise if include_cart_velocity else obs_noise[[0]], action)
                action = np.clip(action, env.min_action, env.max_action)
            total_reward += reward

            for callback in frame_callbacks:
                callback(**dict(agent=aif_agent,
                                env=env,
                                episode_reward=reward,
                                episode_history=episode_history,
                                observations_mapper=observations_mapper,
                                frame=frame))

            if done:
                iterator.set_postfix_str(f'reward={total_reward:.2f}')
                break
            elif i == max_episode_steps - 1:
                iterator.set_postfix_str(f'reward={total_reward:.2f}')

        for callback in episode_callbacks:
            callback(**dict(agent=aif_agent,
                            env=env,
                            episode_reward=total_reward,
                            episode_history=episode_history,
                            observations_mapper=observations_mapper))

        if total_reward > 0:
            aif_agent.learn_biased_model()  # Train biased model with the successful trajectory
        if save_dirpath is not None and train_parameters:
            torch.save(aif_agent.state_dict(), model_save_filepath if not save_all_episodes else os.path.join(save_dirpath, f'model{model_id or ""}_{episode + 1:03d}.pt'))
        VFE, expected_FEEF = aif_agent.logged_history.select_features(['VFE', 'expected_FEEF'])[1]
        training_history.log(episode, 'cumulative_VFE', sum(VFE).item())
        training_history.log(episode, 'cumulative_FEEF', sum(expected_FEEF).item())
        training_history.log(episode, 'steps_per_episode', len(episode_history.times))
        training_history.log(episode, 'rewards', total_reward)

    env.close()
    return training_history


if __name__ == '__main__':
    _display_plots = False
    _load_existing = False
    experiment_dir = './experiments/single_run/'
    _learn_biased_model = True
    _include_cart_velocity = True
    _observation_noise_std = 0.1
    _time_compression = 6
    _planning_horizon = 5  # Multiply with _time_compression to get in simulation steps

    if _learn_biased_model:
        biased_model = BiasedModelBellman(observation_dim=2 if _include_cart_velocity else 1, iterate_train=10, discount_factor=0.995)
    else:
        biased_model = distr.Normal(torch.tensor([0.9, 0.0]) if _include_cart_velocity else 0.9, 1.0)

    from time import time
    t0 = time()
    res = run_training(
        agent_parameters=dict(
            vae=DenseObservation_VAE(
                observation_dim=2 if _include_cart_velocity else 1,
                latent_dim=2 if _include_cart_velocity else 1,
                observation_noise_std=_observation_noise_std),
            biased_model=biased_model,
            policy_dim=1,
            time_step_size=_time_compression,
            planning_horizon=_planning_horizon,
            n_policy_samples=700,
            policy_iterations=2,
            n_policy_candidates=70,
            action_window=2,
            # disable_kl_extrinsic=True,  # Uncomment for ablation study
            # disable_kl_intrinsic=True   # Uncomment for ablation study
        ),
        time_compression=_time_compression,
        episodes=200,
        observation_noise_std=_observation_noise_std,
        include_cart_velocity=_include_cart_velocity,
        model_id=None,
        episode_callbacks=[plots.show_FEEF_vs_FE, plots.show_phase_portrait, plots.show_prediction_vs_outcome] if _display_plots else [],
        frame_callbacks=(),
        save_dirpath=experiment_dir,
        save_all_episodes=False,
        model_load_filepath='./experiments/single_run/model.pt' if _load_existing else None,
        load_vae=True,
        load_transition_model=True,
        load_biased_model=True,
        train_parameters=True,
        verbose=True,
        display_simulation=False
    )

    print(f'Finished {len(res.times)} episodes in {time() - t0:.1f} seconds')
    plots.plot_cumulative_free_energies(res)
    # with open(os.path.join(experiment_dir, 'results.pickle'), 'wb') as f:
    #     pickle.dump(res, f)
    # plots.plot_training_history(res, save_path=os.path.join(experiment_dir, 'run_stats.pdf'), show=True)
