import os
import sys
import json

import gym
import torch
import torch.distributions as distr
import numpy as np
import tqdm

from models.active_inference_capsule import ActiveInferenceCapsule
from utils.timeline import Timeline
from utils.value_map import ValueMap
from utils.model_saving import load_capsule_parameters
from models.vae_dense_obs import DenseObservation_VAE
from models.prior_model import PriorModelBellman
import mountain_car.plotting as plots


def args_to_simulation_settings(args):
    with open(args.settings, 'r') as f:
        settings = json.load(f)

    if 'PriorModelBellman' in settings['agent']['prior_model']:
        prior_model = PriorModelBellman(**settings['agent']['prior_model']['PriorModelBellman'])
    elif 'Normal' in settings['agent']['prior_model']:
        prior_model = distr.Normal(torch.tensor(settings['agent']['prior_model']['Normal']['loc']), torch.tensor(settings['agent']['prior_model']['Normal']['std']))
    else:
        raise KeyError("Unknown prior_model. Make sure it's either PriorModelBellman or Normal")

    settings['agent']['prior_model'] = prior_model
    settings['agent']['vae'] = DenseObservation_VAE(**settings['agent']['vae'])
    settings['simulation']['episode_callbacks'] = [plots.show_phase_portrait, plots.show_prediction_vs_outcome] if args.display_plots else []
    if args.load_existing:
        if args.model_load_filepath != '':
            filepath = args.model_load_filepath
        else:
            filepath = os.path.join(args.save_dirpath, f'model_{settings["experiment_name"]}.pt')

        if os.path.exists(filepath):
            settings['simulation']['model_load_filepath'] = filepath
        else:
            raise FileNotFoundError(f'Previous model <{filepath}> not found.')
    else:
        settings['simulation']['model_load_filepath'] = None
    settings['simulation']['save_dirpath'] = None if args.save_dirpath == '' else args.save_dirpath
    settings['simulation']['model_name'] = settings['experiment_name']
    settings['simulation']['save_all_episodes'] = args.save_all_episodes
    settings['simulation']['verbose'] = args.verbose
    settings['simulation']['display_simulation'] = args.display_simulation
    return settings


def make_model_filepath(dirpath, name, instance=None, episode=None):
    return os.path.join(dirpath, "model_{}{}{}.pt".format(name,
                                                          f"_id{instance}" if instance is not None else '',
                                                          f"_ep{episode:03d}" if episode is not None else ''))


def run_training(agent_parameters,
                 time_compression,
                 observation_noise_std=None,
                 include_cart_velocity=True,
                 model_id=None,
                 hot_start_episodes=0,
                 episodes=1,
                 episode_callbacks=(),
                 frame_callbacks=(),
                 save_dirpath=None,
                 model_name='',
                 model_load_filepath=None,
                 save_all_episodes=False,
                 load_vae=True,
                 load_transition_model=True,
                 load_prior_model=True,
                 train_parameters=True,
                 verbose=True,
                 display_simulation=False):

    env = gym.make('MountainCarContinuous-v0').env
    observations_mapper = ValueMap(in_min=torch.tensor((-1.2, -0.07)), in_max=torch.tensor((0.6, 0.07)),
                                   out_min=torch.tensor((-1.0, -1.0)), out_max=torch.tensor((1.0, 1.0)))
    rewards_mapper = ValueMap(in_min=-100, in_max=100, out_min=-1, out_max=1)
    aif_agent = ActiveInferenceCapsule(**agent_parameters)

    # Load previous model
    if model_load_filepath is not None:
        load_capsule_parameters(aif_agent, model_load_filepath, load_vae, load_transition_model, load_prior_model)
        if verbose:
            loaded_models = []
            loaded_models += ['vae'] if load_vae else []
            loaded_models += ['transition_model'] if load_transition_model else []
            loaded_models += ['prior_model'] if load_prior_model and not isinstance(aif_agent.prior_model, distr.Normal) else []
            print(f"\nLoaded <{', '.join(loaded_models)}> from previous save at <{model_load_filepath}>")

    if save_dirpath is not None and train_parameters:
        torch.save(aif_agent.state_dict(), make_model_filepath(save_dirpath, model_name, model_id, 0 if save_all_episodes else None)) # save episode 0 (no training yet)

    use_kl_intrinsic = aif_agent.use_kl_intrinsic
    use_kl_extrinsic = aif_agent.use_kl_extrinsic
    if not train_parameters:
        aif_agent.eval()
    training_history = Timeline()
    max_episode_steps = 1000
    for episode in range(episodes):
        env.reset()
        aif_agent.reset_states()
        if episode < hot_start_episodes:
            aif_agent.use_kl_intrinsic = True
            aif_agent.use_kl_extrinsic = True
        else:
            aif_agent.use_kl_intrinsic = use_kl_intrinsic
            aif_agent.use_kl_extrinsic = use_kl_extrinsic
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
            reward = rewards_mapper(reward)
            episode_history.log(t, 'true_observations', observation)
            episode_history.log(t - 1, 'true_actions', action)
            obs_noise = observation if observation_noise_std is None else observation + torch.normal(0.0, torch.tensor(observation_noise_std))
            episode_history.log(t, 'noisy_observations', obs_noise)
            if i % time_compression == 0:
                action = aif_agent.step(t, obs_noise if include_cart_velocity else obs_noise[[0]], action, reward)
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
                if i % time_compression != 0:  # if last state did not fall in the update, update anyways
                    aif_agent.step(t, obs_noise if include_cart_velocity else obs_noise[[0]], action=None, reward=reward)
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

        if save_dirpath is not None and train_parameters:
            torch.save(aif_agent.state_dict(), make_model_filepath(save_dirpath, model_name, model_id, episode if save_all_episodes else None))
        VFE, expected_FEEF = aif_agent.logged_history.select_features(['VFE', 'expected_FEEF'])[1]
        training_history.log(episode, 'cumulative_VFE', sum(VFE).item())
        training_history.log(episode, 'cumulative_FEEF', sum(expected_FEEF).item())
        training_history.log(episode, 'steps_per_episode', len(episode_history.times))
        training_history.log(episode, 'rewards', total_reward)

    env.close()
    return training_history
