from typing import List, Union
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch

from models.active_inference_capsule import ActiveInferenceCapsule
from utils.timeline import Timeline


def _plot_observations_actions(axis, agent: ActiveInferenceCapsule, merged_history: Timeline):
    # 1) Plot policy
    times_act_loc, (act_loc, act_std) = merged_history.select_features(['actions_loc', 'actions_std'])
    act_loc, act_std = torch.stack(act_loc).view(-1), torch.stack(act_std).view(-1)
    axis.fill_between(times_act_loc, act_loc - act_std, act_loc + act_std, color='r', alpha=0.3, linewidth=0)
    pl_pol = axis._display_plots(times_act_loc, act_loc, 'r--', linewidth=1, label='Policy')

    # 2) Plot executed actions
    times_actions, true_actions = merged_history.select_features('true_actions')
    pl_act = axis._display_plots(times_actions, true_actions, 'b', linewidth=1, label='Executed action')

    axis.set_ylabel('a', color='r', rotation=0)
    axis.tick_params(axis='y', labelcolor='r')
    axis_obs = axis.twinx()

    # 3) Plot true observations
    times_true_observations, true_observations = merged_history.select_features('true_observations')
    times_noisy_observations, noisy_observations = merged_history.select_features('noisy_observations')
    true_observations = torch.stack(true_observations)
    noisy_observations = torch.stack(noisy_observations)
    pl_pos = axis_obs._display_plots(times_true_observations, true_observations[:, 0], color='k', linewidth=1.0, label='true position')
    pl_pos_noise = axis_obs._display_plots(times_noisy_observations, noisy_observations[:, 0], color='k', linestyle='--', linewidth=1.0, label='noisy observation')
    # 4) Plot expected observations
    times_filtered, (filtered_locs, filtered_stds) = merged_history.select_features(['filtered_observations_locs', 'filtered_observations_stds'])
    locs_, stds_ = torch.stack(filtered_locs)[:, 0], torch.stack(filtered_stds)[:, 0]
    axis_obs.fill_between(times_filtered, locs_ + stds_, locs_ - stds_, color='k', alpha=0.3)
    pl_rec = axis_obs._display_plots(times_filtered, locs_, 'k', linestyle='dotted', linewidth=1.0, label='filtered belief')
    axis_obs.set_ylabel('x', rotation=0)  # we already handled the x-label with ax1

    # Make common legend for both axes
    lns = pl_act + pl_pol + pl_pos + pl_pos_noise + pl_rec
    labs = [ln.get_label() for ln in lns]
    axis.legend(lns, labs, loc='lower right', framealpha=0.4)
    axis.set_ylim((-2.5, 2.5))
    axis.set_yticks([-2, -1, 0, 1, 2])
    axis_obs.set_ylim((-1.1, 1.1))
    axis.grid(linewidth=0.5, alpha=0.5)
    axis.set_title('Policy and car position')

    return axis


def _plot_latent_prediction(axis, latent_idx, merged_history: Timeline):
    # 1) plot prediction tubes
    times_replanning, predictions = merged_history.select_features('predictions')
    for j, prediction in enumerate(predictions):
        times_pred, (pred_locs, pred_stds) = prediction.select_features(['pred_locs', 'pred_stds'])
        pred_locs, pred_stds = torch.stack(pred_locs)[:, latent_idx], torch.stack(pred_stds)[:, latent_idx]  # convert to tensor and select latent dimension i
        axis.fill_between(times_pred, pred_locs - pred_stds, pred_locs + pred_stds, color='k', alpha=0.1, linewidth=1)
        axis._display_plots(times_pred, pred_locs, 'k--', alpha=0.7, linewidth=0.3, label='Predicted latent' if j == 1 else None)

    # 2) plot expected latents
    times_expectations, (expected_locs, expected_stds) = merged_history.select_features(['expected_locs', 'expected_stds'])
    expected_locs, expected_stds = torch.stack(expected_locs)[:, latent_idx], torch.stack(expected_stds)[:, latent_idx]  # convert to tensor and select latent dimension i
    axis.fill_between(times_expectations, expected_locs - expected_stds, expected_locs + expected_stds, color='r', linewidth=0, alpha=0.3)
    axis._display_plots(times_expectations, expected_locs, color=(0.8, 0, 0), linestyle='--', linewidth=1.0, alpha=0.8, label='Expected latent')

    # 3) Plot perceived latents
    times_percepts, (perceived_locs, perceived_stds) = merged_history.select_features(['perceived_locs', 'perceived_stds'])
    perceived_locs, perceived_stds = torch.stack(perceived_locs)[:, latent_idx], torch.stack(perceived_stds)[:, latent_idx]  # convert to tensor and select latent dimension i
    axis.fill_between(times_percepts, perceived_locs - perceived_stds, perceived_locs + perceived_stds, color='b', linewidth=0, alpha=0.4)
    axis._display_plots(times_percepts, perceived_locs, 'b', linewidth=1.0, label='Perceived latent')

    axis.grid(linewidth=0.5, alpha=0.5)
    axis.set_title(f'Latent {latent_idx + 1}')
    axis.legend(loc='lower right', framealpha=0.3)


def _plot_phase_portrait(fig, axis, agent: ActiveInferenceCapsule, episode_history: Timeline, observations_mapper, **kwargs):
    axis.set_aspect(1.0)
    grid_points = 30
    # 1) Plot heat map of extrinsic KL divergence
    sample_positions = torch.linspace(-1.0, 1.0, grid_points)  # positions in the agent space (-1.2, 0.6) -> (-1.0, 1.0)
    sample_velocities = torch.linspace(-1.0, 1.0, grid_points)  # positions in the agent space (-1.2, 0.6) -> (-1.0, 1.0)
    kl_extrinsic = torch.zeros((grid_points, grid_points))
    with torch.no_grad():
        for i in range(grid_points):
            for j in range(grid_points):
                if agent.observation_dim == 1:  # Case where the agent cannot observe the velocity
                    kl_extrinsic[i, j] = agent.kl_extrinsic(sample_positions[[j]]).sum()
                else:
                    kl_extrinsic[i, j] = agent.kl_extrinsic(torch.stack((sample_positions[j], sample_velocities[i]))).sum()

    clev = torch.linspace(kl_extrinsic.min().item(), kl_extrinsic.max().item(), 100)
    cs = axis.contourf(sample_positions.expand((grid_points, grid_points)), sample_velocities.expand((grid_points, grid_points)).transpose(1, 0), kl_extrinsic, clev, cmap='magma')

    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(cs, cax=cax, ticks=torch.linspace(kl_extrinsic.min().item(), kl_extrinsic.max().item(), 6))
    cbar.set_label('KL extr.')

    # 2) Plot trajectories
    true_observations = torch.stack(episode_history.select_features('true_observations')[1])
    axis._display_plots(true_observations[:, 0], true_observations[:, 1], color=(0.5, 0.5, 1.0), linewidth=1.5)
    axis.scatter([true_observations[0, 0]], [true_observations[0, 1]], color='r')

    if agent.observation_dim == 2:
        perceived_locs = agent.logged_history.select_features('perceived_locs')[1]
        if len(perceived_locs) > 1:  # Skip if no planning done yet
            perceived_locs = torch.stack(perceived_locs)
            perceived_observations = agent.vae.decode(perceived_locs).detach()
            axis._display_plots(perceived_observations[:, 0], perceived_observations[:, 1], color=(0.0, 1.0, 0.0), linestyle='--', linewidth=1.5)

        # Plot predicted trajectories
        times_predictions, predictions = agent.logged_history.select_features('predictions')
        for j, prediction in enumerate(predictions):
            _, pred_locs = prediction.select_features('pred_locs')
            pred_locs = torch.stack(pred_locs)
            perceived_locs = agent.logged_history.get_frame(times_predictions[j])['perceived_locs']
            # pred_locs = torch.stack(pred_locs)
            pred_obs = agent.vae.decode(torch.cat([perceived_locs.unsqueeze(0), pred_locs])).detach()
            axis._display_plots(pred_obs[:, 0], pred_obs[:, 1], color=(0.0, 1.0, 0.0), alpha=0.5, linewidth=0.5, label='Perceived latent' if j == 1 else None)

    # 3) Plot goal box
    x1, y1 = observations_mapper(torch.tensor((0.45, 0.0)))  # thresholds for mountain-car goal, transformed to problem coordinates
    x2, y2 = 1.0, 1.0
    axis.add_patch(patches.Polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], edgecolor='k', facecolor=(0.0, 1.0, 0.0), linestyle='--', alpha=0.5))

    axis.set_xlabel('Horizontal position')
    axis.set_ylabel('Velocity', labelpad=-3)
    axis.set_title('Phase portrait with extrinsic value')
    axis.set_xlim([-1, 1])
    axis.set_ylim([-1, 1])
    axis.grid(color=(0.5, 0.5, 0.5), alpha=0.5, linewidth=0.2)
    return axis


def show_prediction_vs_outcome(agent: ActiveInferenceCapsule, episode_history: Timeline, **kwargs):
    fig = plt.figure(figsize=(8, 6))
    merged_history = episode_history.merge(agent.logged_history)

    # 1) Plot actions and states
    ax2 = fig.add_subplot(agent.latent_dim + 1, 1, 1)
    _plot_observations_actions(ax2, agent, merged_history)

    for i in range(agent.latent_dim):
        ax1 = fig.add_subplot(agent.latent_dim + 1, 1, i + 2)
        _plot_latent_prediction(ax1, i, merged_history)
        ax2.set_xlim(*ax1.get_xlim())  # Make sure the timelines match between the action-state plot and the latent-predictions plots

    fig.tight_layout()
    plt.show()
    plt.close()


def show_phase_portrait(agent: ActiveInferenceCapsule, episode_history: Timeline, observations_mapper, **kwargs):
    fig = plt.figure(figsize=(6, 4))
    axis = fig.gca()
    _plot_phase_portrait(fig, axis, agent, episode_history, observations_mapper)
    plt.show()
    plt.close()


def show_FEEF_vs_FE(agent: ActiveInferenceCapsule, **kwargs):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.gca()
    times_FEEF, expected_FEEF = agent.logged_history.select_features('expected_FEEF')
    times_FE, VFE = agent.logged_history.select_features('VFE')
    expected_FEEF = torch.stack(expected_FEEF).view(-1)
    VFE = torch.stack(VFE).view(-1)
    ax._display_plots(times_FE, VFE, 'b-', label='VFE')
    ax._display_plots(times_FEEF, expected_FEEF, 'r--', label='FEEF')
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close()


def plot_training_history(timelines: Union[Timeline, List[Timeline]], save_path=None, show=True, figure=None, label=None, color=(0.2, 0.4, 1.0)):
    timelines = [timelines] if isinstance(timelines, Timeline) else timelines
    all_rewards = []
    times = None
    for timeline in timelines:
        times, rewards = timeline.select_features('steps_per_episode')
        all_rewards.append(rewards)
    all_rewards = torch.tensor(all_rewards, dtype=torch.float32)

    r_mean = all_rewards.mean(0)
    r_max = r_mean + all_rewards.std(0) # all_rewards.max(0)[0]
    r_min = [max(a, b) for a, b in zip(r_mean - all_rewards.std(0), all_rewards.min(0)[0])]

    fig = figure or plt.figure(figsize=(6, 4))
    ax = fig.gca()
    if len(all_rewards) > 1:
        ax.fill_between(times, r_min, r_max, color=color, linewidth=0, alpha=0.3)
    ax._display_plots(times, r_mean, color=color, label=label)
    ax.set_yticks([0, min(r_min), 200, 400, 600, 800, 1000])
    ax.set_ylim((0, 1000.0))

    if figure is None:
        plt.grid(linewidth=0.4, alpha=0.5)
        plt.axhline(200, color='k', linestyle='--', linewidth=0.5)
        plt.suptitle(f'Mountain car. Statistics of {len(timelines)} agents', y=0.94)
        plt.xlabel('Episodes')
        plt.ylabel('Steps until goal')
        if os.path.exists(os.path.dirname(save_path)):
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()
    else:
        return fig


def plot_training_free_energy(timelines: Union[Timeline, List[Timeline]], save_path=None, show=True, figure=None, label=None, color=(0.2, 0.4, 1.0)):
    timelines = [timelines] if isinstance(timelines, Timeline) else timelines
    all_free_energies = []
    times = None
    for timeline in timelines:
        times, free_energy = timeline.select_features('cumulative_VFE')
        all_free_energies.append(free_energy)
    all_free_energies = torch.tensor(all_free_energies, dtype=torch.float32)

    r_mean = all_free_energies.mean(0)
    # r_max = r_mean + all_free_energies.std(0) # all_rewards.max(0)[0]
    r_max = [min(a, b) for a, b in zip(r_mean + all_free_energies.std(0), all_free_energies.max(0)[0])]
    r_min = [max(a, b) for a, b in zip(r_mean - all_free_energies.std(0), all_free_energies.min(0)[0])]

    fig = figure or plt.figure(figsize=(6, 4))
    ax = fig.gca()
    if len(all_free_energies) > 1:
        ax.fill_between(times, r_min, r_max, color=color, linewidth=0, alpha=0.3)
    ax._display_plots(times, r_mean, color=color, label=label)

    if figure is None:
        plt.grid(linewidth=0.4, alpha=0.5)
        plt.axhline(200, color='k', linestyle='--', linewidth=0.5)
        plt.suptitle(f'Mountain car. Statistics of {len(timelines)} agents', y=0.94)
        plt.xlabel('Episodes')
        plt.ylabel('Episode cumulative free energy')
        if os.path.exists(os.path.dirname(save_path)):
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()
    else:
        return fig


def plot_cumulative_free_energies(timeline: Timeline):
    episodes, (cumulative_VFE, cumulative_FEEF) = timeline.select_features(['cumulative_VFE', 'cumulative_FEEF'])
    fig = plt.figure(figsize=(6, 4))
    ax = fig.gca()
    ax._display_plots(episodes, cumulative_VFE, 'b-', label='Cumulative FE')
    ax._display_plots(episodes, cumulative_FEEF, 'r--', label='Cumulative FEEF')
    ax.legend()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Free energy')
    plt.show()
    plt.close()


def make_video_frame(agent: ActiveInferenceCapsule, episode_history: Timeline, render, observations_mapper):
    fig = plt.figure(figsize=(6, 4.5))
    gs = fig.add_gridspec(5, 2)
    ax_phase = fig.add_subplot(gs[:3, 0])
    ax_frame = fig.add_subplot(gs[:3, 1])
    ax_action = fig.add_subplot(gs[3:, :])

    _plot_phase_portrait(fig, ax_phase, agent, episode_history, observations_mapper)
    ax_phase.set_xticks([-1, -0.5, 0, 0.5, 1.0])
    ax_phase.set_yticks([-1, -0.5, 0, 0.5, 1.0])

    ax_frame.set_title('Simulation frame')
    ax_frame.imshow(render)
    ax_frame.get_xaxis().set_visible(False)
    ax_frame.get_yaxis().set_visible(False)

    ax_action = _plot_observations_actions(ax_action, agent, episode_history.merge(agent.logged_history))
    ax_action.set_xlim((0, 1000))
    fig.tight_layout()

    # plt.show()
    return fig


if __name__ == '__main__':
    import pickle

    with open('./experiments/batch_run/results.pickle', 'rb') as f:
        tmlns = pickle.load(f)
    plot_training_history(tmlns, save_path='./experiments/batch_run/run_stats.pdf', show=True)
