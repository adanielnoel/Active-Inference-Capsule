import sys
import os
import subprocess
import glob
import pickle

from tqdm import tqdm
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.active_inference_capsule import ActiveInferenceCapsule
from utils.model_saving import load_capsule_parameters
from utils.value_map import ValueMap
from utils.timeline import Timeline
import mountain_car.plotting as plots

model_path = './experiments/single_run/model.pt'
save_dir = './experiments/single_run/'

if __name__ == '__main__':
    # Load model settings and set up agent and simulation
    with open(os.path.join(os.path.dirname(model_path), 'settings.pk'), 'rb') as f:
        settings = pickle.load(f)
    aif_agent = ActiveInferenceCapsule(**settings['agent_parameters'])
    load_capsule_parameters(aif_agent, model_path, load_vae=True, load_transition_model=True, load_biased_model=True)
    env = gym.make('MountainCarContinuous-v0').env
    observations_mapper = ValueMap(in_min=torch.tensor((-1.2, -0.07)), in_max=torch.tensor((0.6, 0.07)),
                                   out_min=torch.tensor((-1.0, -1.0)), out_max=torch.tensor((1.0, 1.0)))
    time_compression = settings['time_compression']
    observation_noise_std = settings['observation_noise_std']

    # Prepare save paths
    model_name = os.path.basename(model_path).split(".")[0]
    frames_path = os.path.join(save_dir, f'frames_{model_name}')
    video_path = os.path.join(save_dir, f'video_{model_name}.mp4')
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)   # Make a directory to save the frame images

    # If a video was previously made for the same model, delete it and clear all frames
    for file_name in glob.glob(os.path.join(frames_path, '*.png')):
        os.remove(file_name)
    if os.path.exists(video_path):
        os.remove(video_path)

    aif_agent.eval()    # Do not train parameters
    action = aif_agent.step(0, observations_mapper(torch.tensor(env.state, dtype=torch.float32))) # get first action
    episode_history = Timeline()
    max_episode_steps = 1000
    iterator = tqdm(range(max_episode_steps), file=sys.stdout)
    for i in iterator:
        img = env.render(mode='rgb_array')
        t = i + 1
        observation, _, done, _ = env.step(action)
        observation = observations_mapper(torch.from_numpy(observation).float())
        episode_history.log(t, 'true_observations', observation)
        episode_history.log(t - 1, 'true_actions', action)
        if i % time_compression == 0:
            obs_noise = observation if observation_noise_std is None else observation + np.random.normal(loc=0.0, scale=observation_noise_std)
            action = aif_agent.step(t, obs_noise, action)
            action = np.clip(action, env.min_action, env.max_action)

        fig = plots.make_video_frame(aif_agent, episode_history, img, observations_mapper)
        fig.savefig(os.path.join(frames_path, f'frame_{i:04d}.png'), dpi=200)
        plt.close(fig)

        if done:
            break

    env.close()

    subprocess.call([
        'ffmpeg',
        '-i', os.path.join(frames_path, 'frame_%04d.png'),  # input images
        '-r', '25',  # output frame rate
        '-pix_fmt', 'yuv420p',
        '-b', '5000k',  # 5Mb bitrate
        video_path
    ])
