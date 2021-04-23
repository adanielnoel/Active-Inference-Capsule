# Active Inference Capsule

## Requirements
- Python 3.5+
- Pytorch (tested on 1.4.0)
- numpy
- gym
- tqdm
- matplotlib
- ffmpeg (only if generating videos)

---

## Training a single agent
The training routine is implemented in the script `mountain_car/training.py`. The parameters are at the bottom of the file. This file contains the `run_training()` routine which is used by other scripts. Only modify the parameters after `if __name__ == '__main__'`.

- To train a new agent, set `_display_plots = False` and `_load_existing = False`. This will save the model after every episode, typically in `mountain_car/experiments/single_run/model.pt`.
- To plot results, set `_display_plots = True` and `_load_existing = True` and run again.

## Training a batch of agents
Many independent agents can be trained simultaneously with the script `mountain_car/batch_training.py`. By default, this will try to use all available CPUs, which can be changed setting `max_cpus = #`.
The routine spawns multiple simulations in parallel and saves the results to a `.pickle` file. Interesting parameter sets can be found in the file `mountain_car/parameter_sets.txt`. The results of these parameter sets are already saved in `mountain_car/experiments/batch_run/` for convenience.

- To train a batch of agents, set the desired parameters and run the script. This will save the results after every agent that finishes training, typically in `mountain_car/experiments/batch_run/`.
- To plot training statistics, run the script `mountain_car/plot_multiple_trainings.py`. In it, you can specify which results to plot. 

## Generating a video
- First, train a single agent. Note: in `mountain_car/training.py` you can specify `save_all_episodes=True`, which will save a model per episode instead of overriding the previous one.
- Specify the path to the desired model in `mountain_car/make_video.py` and run it. This will start saving individual frames, typically in a subdirectory in `mountain_car/experiments/single_run/`. When the agent reaches the goal, it will call `ffmpeg` to generate an `mp4` file.

