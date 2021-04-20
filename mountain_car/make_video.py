import os
import subprocess
import glob
import shutil
import pickle

import matplotlib.pyplot as plt

import mountain_car.plotting as plots
from mountain_car.training import run_training

model_path = './experiments/single_run/model.pt'
save_dir = './experiments/single_run/'
freeze_end = True  # Whether to add 1s of video at the last frame

# Prepare save paths
model_name = os.path.basename(model_path).split(".")[0]
frames_path = os.path.join(save_dir, f'frames_{model_name}')
video_path = os.path.join(save_dir, f'video_{model_name}.mp4')
if not os.path.exists(frames_path):
    os.makedirs(frames_path)  # Make a directory to save the frame images


# Callback function to draw each episode frame
def save_video_frame(agent, episode_history, observations_mapper, frame, **kwargs):
    fig = plots.make_video_frame(agent, episode_history, frame, observations_mapper)
    if model_name[-4] == '_':
        fig.subplots_adjust(top=0.88)
        fig.suptitle(f'Episode {int(model_name[-3:])}', y=0.98)
    fig.savefig(os.path.join(frames_path, f'frame_{len(episode_history.times):04d}.png'), dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    # If a video was previously made for the same model, delete it and clear all frames
    for file_name in glob.glob(os.path.join(frames_path, '*.png')):
        os.remove(file_name)
    if os.path.exists(video_path):
        os.remove(video_path)

    # Load model and simulation settings
    with open(os.path.join(os.path.dirname(model_path), 'settings.pk'), 'rb') as f:
        settings = pickle.load(f)

    # Run the episode, rendering and saving each frame
    run_training(
        **settings,
        frame_callbacks=[save_video_frame],
        display_simulation=True,
        train_parameters=True,
        model_load_filepath=model_path
    )

    if freeze_end:  # Freeze the last second of video by repeating the last frame fps times
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
