# Active Inference Capsule
Repository for the paper **Online reinforcement learning with sparse rewards through an active inference capsule**, submitted to NeurIPS2021.

<img src="https://user-images.githubusercontent.com/33813179/119155212-38a01800-ba53-11eb-880c-1a6e65974b8f.gif" width="400">


## Requirements
- `Python` 3.5+
- `Pytorch` (tested on 1.4.0)
- `numpy`
- `gym`
- `tqdm`
- `matplotlib`
- `ffmpeg` (only if generating videos)

---

## Training a single agent
```bash
python run_capsule_mountain_car.py --settings='./paper_results/settings_learned_prior_H5.json' --save_dirpath='./paper_results/simulation_results/'
```
This will train a single agent and save the model in `./paper_results/simulation_results/`

By default, the training progress will be printed. There are setting files for other agents in `./paper_results`

Note: the folder `./paper_results/simulation_results/` already contains the results used in the paper.

## Training a batch of agents
```bash
python run_capsule_mountain_car.py --settings='./paper_results/settings_learned_prior_H5.json' --save_dirpath='./paper_results/simulation_results/' --batch_agents=30
```
This will train 30 agents and save the models and training statistics in `./paper_results/simulation_results/`

By default, this will use all available CPU cores. It can be changed by the option `--max_cpu`.

The results can be visualized using the script `./mountain_car/plot_multiple_trainings.py` or by re-making the figures with `./paper_results/make_figures.py` (this will also generate other figures than the training plots).

## Generating a video
First, train a single agent. With the option `--save_all_episodes=True` when training the single agent, the model is saved per episode instead of overriding the latest one. This facilitates making videos of different episodes of the same agent.

```bash
python run_capsule_mountain_car.py --settings='./paper_results/settings_learned_prior_H5.json' --save_dirpath='./paper_results/simulation_results/' --make_video=True --model_load_filepath='./paper_results/simulation_results/model_learned_prior_H5.pt'
```

This will start saving individual frames in a subdirectory of `save_dirpath`. When the agent reaches the goal, it will call `ffmpeg` to generate a `.mp4` file.

This process is much slower than training because `matplotlib` takes quite some time to render each frame.

## Other
A full list of command-line options can be found in `run_capsule_mountain_car.py`

