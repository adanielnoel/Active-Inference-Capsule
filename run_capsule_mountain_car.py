import os
import re
import sys
import copy
import glob
import shutil
import pickle
import argparse
import subprocess
from time import time
import multiprocessing

from tqdm import tqdm
import matplotlib.pyplot as plt

from mountain_car.training import args_to_simulation_settings
from mountain_car.training import run_training
import mountain_car.plotting as plots


parser = argparse.ArgumentParser()
parser.add_argument("--settings", type=str, default="./paper_results/settings_learned_prior_H5.json")
parser.add_argument("--batch_agents", type=int, default=1)
parser.add_argument("--max_cpu", type=int, default=-1)
parser.add_argument("--make_video", type=bool, default=False)
parser.add_argument("--display_plots", type=bool, default=False)
parser.add_argument("--load_existing", type=bool, default=False)
parser.add_argument("--save_dirpath", type=str, default='./paper_results/simulation_results/')
parser.add_argument("--model_load_filepath", type=str, default='')
parser.add_argument("--save_all_episodes", type=bool, default=False)
parser.add_argument("--verbose", type=bool, default=True)
parser.add_argument("--display_simulation", type=bool, default=False)
parser.add_argument("--video_with_dashboard", type=bool, default=False)
args = parser.parse_args()
args.load_existing = args.make_video or args.load_existing # make sure we are loading a model when making a video
settings = args_to_simulation_settings(args)


# ------- TRAINING A SINGLE AGENT ---------
def train_single_agent():
    global settings
    global args
    run_training(
        agent_parameters=settings['agent'],
        **settings['simulation']
    )


def _run_training_process(training_id):
    global settings
    global args
    settings_copy = copy.deepcopy(settings)
    settings_copy['simulation']['verbose'] = False
    settings_copy['simulation']['model_id'] = training_id
    return run_training(agent_parameters=settings_copy['agent'], **settings_copy['simulation'])


# ------- TRAINING A BATCH OF AGENTS ---------
def train_many_agents():
    global settings
    global args
    num_cpu_cores = args.max_cpu if args.max_cpu > 0 else multiprocessing.cpu_count()
    num_processes = min(args.batch_agents, num_cpu_cores)
    if not os.path.exists(args.save_dirpath):
        os.makedirs(args.save_dirpath)

    print(f'\nRunning {args.batch_agents} simulations of {settings["experiment_name"]} in {num_processes} parallel processes...')
    t0 = time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        all_results = []
        for new_result in tqdm(pool.imap_unordered(_run_training_process, range(args.batch_agents)), total=args.batch_agents, file=sys.stdout):
            all_results.append(new_result)
            with open(os.path.join(args.save_dirpath, f'results_{settings["experiment_name"]}.pickle'), 'wb') as f:
                pickle.dump(all_results, f)
    t1 = time() - t0
    print(f'Finished {settings["simulation"]["episodes"] * args.batch_agents} episodes in {t1/60:.1f} minutes ({t1 / (settings["simulation"]["episodes"] * args.batch_agents):.2f}s/episode)')


# ------- MAKING A VIDEO ---------
def make_video():
    global settings
    global args
    if settings['simulation']['model_load_filepath'] is None or not os.path.exists(settings['simulation']['model_load_filepath']):
        raise RuntimeError('Provide a trained model through model_load_filepath for generating a video')

    frames_path = os.path.join(args.save_dirpath, f'frames_{settings["experiment_name"]}')
    video_path = os.path.join(args.save_dirpath, f'video_{settings["experiment_name"]}.mp4')

    if not os.path.exists(frames_path):
        os.makedirs(frames_path)  # Make a directory to save the frame images

    # If a video was previously made for the same model, delete it and clear all frames
    for file_name in glob.glob(os.path.join(frames_path, '*.png')):
        os.remove(file_name)
    if os.path.exists(video_path):
        os.remove(video_path)

    # Callback function to draw each episode frame
    def save_video_frame(agent, episode_history, observations_mapper, frame, **kwargs):
        if args.video_with_dashboard:
            fig = plots.make_video_frame(agent, episode_history, frame, observations_mapper)
            match = re.search(r'_ep(\d*).pt', settings['simulation']['model_load_filepath'])
            if match:  # the model contains the episode number
                fig.subplots_adjust(top=0.88)
                fig.suptitle(f'Episode {int(match.group(1))}', y=0.98)
        else:
            fig = plt.figure(figsize=(6, 4))
            fig.gca().imshow(frame)
            fig.gca().get_xaxis().set_visible(False)
            fig.gca().get_yaxis().set_visible(False)
            fig.tight_layout()
        fig.savefig(os.path.join(frames_path, f'frame_{len(episode_history.times):04d}.png'), dpi=200)
        plt.close(fig)

    settings['simulation']['observation_noise_std'] = settings['simulation']['observation_noise_std'] or 0.05  # When noise is None, set it to 0.05 for display
    settings['simulation']['frame_callbacks'] = [save_video_frame]
    settings['simulation']['display_simulation'] = True
    settings['simulation']['train_parameters'] = False
    settings['simulation']['episodes'] = 1
    # Run the episode, rendering and saving each frame
    run_training(
        agent_parameters=settings['agent'],
        **settings['simulation']
    )

    # Freeze the last second of video by repeating the last frame fps times
    last_frame = sorted(glob.glob(os.path.join(frames_path, f'frame_*.png')))[-1]
    last_id = int(last_frame[-8:-4])
    for i in range(1, 25):
        shutil.copy(last_frame, os.path.join(frames_path, f'frame_{last_id + i:04d}.png'))

    # Generate video from saved frames
    subprocess.call([
        'ffmpeg',
        '-i', os.path.join(frames_path, 'frame_%04d.png'),  # input images
        '-r', '25',  # output frame rate
        '-pix_fmt', 'yuv420p',
        '-b', '5000k',  # 5Mb bitrate
        video_path
    ])


if __name__ == "__main__":
    if args.batch_agents == 1 and args.make_video is False:
        train_single_agent()
    elif args.batch_agents > 1 and args.make_video is False:
        train_many_agents()
    elif args.batch_agents == 1 and args.make_video is True:
        make_video()
    elif args.batch_agents < 1:
        raise RuntimeError('At least one agent requires (batch_agents=1)')
    else:
        raise RuntimeError('Cannot make video with batch_agents != 1')
