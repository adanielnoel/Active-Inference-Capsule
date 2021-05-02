import pickle
import os

import matplotlib.pyplot as plt

from mountain_car.plotting import plot_training_history, plot_training_free_energy, _plot_phase_portrait, _plot_observations_actions, _plot_latent_prediction
from mountain_car.training import run_training
from models.active_inference_capsule import ActiveInferenceCapsule
from utils.timeline import Timeline


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})


def goal_vs_rewards():
    results = [
        ('Random agent', (0.7, 0.7, 0.7), 'dotted', 'results_random.pickle'),
        ('Given prior, T=60', (0.8, 0.3, 0.3), '-.', 'results_normal60.pickle'),
        ('Given prior, T=90', (0.3, 0.5, 1.0), '--', 'results_normal90.pickle'),
        ('Learned prior, T=36', (0.3, 0.7, 0.3), '-', 'results_bellman36.pickle'),
    ]

    fig_st = plt.figure(1, figsize=(6, 3))
    fig_st.gca().grid(linewidth=0.4, alpha=0.5)
    fig_st.gca().axhline(200, color='k', linestyle='--', linewidth=0.5)
    fig_st.gca().set_title(f'Given vs. learned priors (episode length)')
    fig_st.gca().set_xlabel('Episodes')
    fig_st.gca().set_ylabel('Steps until goal')

    for label, color, linestyle, results_file in results:
        with open(results_file, 'rb') as f:
            tmlns = pickle.load(f)
        plot_training_history(tmlns, figure=fig_st, label=label, color=color, linestyle=linestyle)

    fig_st.gca().legend(loc='upper right', framealpha=0.5)
    fig_st.tight_layout()
    fig_st.gca().set_xlim((0, 150))
    fig_st.gca().set_ylim((0, 1000))
    # plt.show()

    fig_st.savefig('./figures/goal_vs_rewards_episode_length.pdf')


def clean_vs_noise():
    results = [
        ('Learned prior, T=36', (0.3, 0.5, 1.0), '-', 'results_bellman36.pickle'),
        ('Learned prior, T=36, with noise', (0.8, 0.3, 0.3), '--', 'results_bellman36_noise.pickle')
    ]

    fig_st = plt.figure(1, figsize=(6, 3))
    fig_st.gca().grid(linewidth=0.4, alpha=0.5)
    fig_st.gca().axhline(200, color='k', linestyle='--', linewidth=0.5)
    fig_st.gca().set_title(f'Clean vs. noisy observations (episode length)')
    fig_st.gca().set_xlabel('Episodes')
    fig_st.gca().set_ylabel('Steps until goal')

    fig_fe = plt.figure(2, figsize=(6, 3))
    fig_fe.gca().grid(linewidth=0.4, alpha=0.5)
    fig_fe.gca().set_title(f'Clean vs. noisy observations (free energy)')
    fig_fe.gca().set_xlabel('Episodes')
    fig_fe.gca().set_ylabel('Episode cumulative free energy')

    for label, color, linestyle, results_file in results:
        with open(results_file, 'rb') as f:
            tmlns = pickle.load(f)
        plot_training_history(tmlns, figure=fig_st, label=label, color=color, linestyle=linestyle)
        plot_training_free_energy(tmlns, figure=fig_fe, label=label, color=color, linestyle=linestyle)

    fig_st.gca().legend(framealpha=0.4)
    fig_fe.gca().legend(framealpha=0.4)
    fig_st.tight_layout()
    fig_fe.tight_layout()
    fig_st.gca().set_xlim((0, 150))
    fig_st.gca().set_ylim((0, 1000))
    fig_fe.gca().set_xlim((0, 150))
    fig_fe.gca().set_ylim((-200, 3000))
    plt.show()

    fig_fe.savefig('./figures/clean_vs_noise_free_energy.pdf')
    fig_st.savefig('./figures/clean_vs_noise_episode_length.pdf')


def phase_portraits_goal_vs_rewards():
    models = [
        ('Given prior, T=60', 'model_normal60.pt', 'settings_normal60.pk'),
        ('Given prior, T=90', 'model_normal90.pt', 'settings_normal90.pk'),
        ('Learned prior, T=36', 'model_bellman36.pt', 'settings_bellman36.pk'),
    ]

    fig = plt.figure(figsize=(8, 2.4))
    axes = []
    for i, (model_name, model_file, settings_file) in enumerate(models):
        axis = fig.add_subplot(1, 3, i + 1)

        with open(os.path.join(settings_file), 'rb') as f:
            settings = pickle.load(f)

        # Run an episode and plot the phase portrait onto the axis
        run_training(
            **settings,
            episode_callbacks=[lambda agent, episode_history, observations_mapper, **kwargs: _plot_phase_portrait(fig, axis, agent, episode_history, observations_mapper, label_cbar=i==2)],
            display_simulation=False,
            train_parameters=False,
            model_load_filepath=model_file,
        )

        axis.set_title(model_name)
        axes.append(axis)
    axes[1].set_ylabel('')
    axes[2].set_ylabel('')
    fig.tight_layout()
    # plt.show()
    plt.savefig('./figures/goal_vs_rewards_phase_portraits.pdf')


def observations_and_policy_noise():
    model_file = 'model_bellman36_noise.pt'
    settings_file = 'settings_bellman36_noise.pk'

    fig = plt.figure(figsize=(7, 2.2))
    ax_obs = fig.gca()
    # gs = fig.add_gridspec(5, 1)
    # ax_obs = fig.add_subplot(gs[:3, 0])
    # ax_lat1 = fig.add_subplot(gs[3, 0])
    # ax_lat2 = fig.add_subplot(gs[4, 0])

    def make_plot(agent: ActiveInferenceCapsule, episode_history: Timeline, **kwargs):
        merged_history = episode_history.merge(agent.logged_history)

        # 1) Plot actions and states
        _plot_observations_actions(ax_obs, agent, merged_history)
        # _plot_latent_prediction(ax_lat1, 0, merged_history)
        # _plot_latent_prediction(ax_lat2, 1, merged_history)

    with open(os.path.join(settings_file), 'rb') as f:
        settings = pickle.load(f)

    # Run an episode and plot the phase portrait onto the axis
    run_training(
        **settings,
        episode_callbacks=[make_plot],
        display_simulation=False,
        train_parameters=False,
        model_load_filepath=model_file,
    )

    ax_obs.set_xlim((0, 120))
    ax_obs.set_xlabel('Steps')
    # ax_lat1.set_xlim((0, 120))
    # ax_lat2.set_xlim((0, 120))
    fig.tight_layout()
    # plt.show()
    plt.savefig('./figures/clean_vs_noise_observations_and_policy.pdf')


if __name__ == '__main__':
    # goal_vs_rewards()
    # phase_portraits_goal_vs_rewards()
    # clean_vs_noise()
    observations_and_policy_noise()
