import pickle

import matplotlib.pyplot as plt

from mountain_car.plotting import plot_training_history, plot_training_free_energy, _plot_phase_portrait, _plot_observations_actions, _plot_latent_prediction
from mountain_car.training import args_to_simulation_settings, run_training
from models.active_inference_capsule import ActiveInferenceCapsule
from utils.timeline import Timeline
from utils.args_class import Args

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})


def goal_vs_rewards():
    results = [
        ('Random agent', (190 / 255, 190 / 255, 190 / 255), 'dotted', './simulation_results/results_random_agent.pickle'),
        ('Given prior, H=15', (255 / 255, 35 / 255, 100 / 255), '--', './simulation_results/results_given_prior_H15.pickle'),
        ('Given prior, H=10', (203 / 255, 161 / 255, 39 / 255), '-.', './simulation_results/results_given_prior_H10.pickle'),
        ('Learned prior, H=10', (100 / 255, 180 / 255, 255 / 255), (0, (3, 1, 1, 1, 1, 1)), './simulation_results/results_learned_prior_H10.pickle'),
        ('Learned prior, H=5', (210 / 255, 100 / 255, 255 / 255), '-', './simulation_results/results_learned_prior_H5.pickle'),
    ]

    fig_st = plt.figure(1, figsize=(6, 2.5))
    ax_st = fig_st.gca()
    ax_st.grid(linewidth=0.4, alpha=0.5)
    ax_st.axhline(200, color='k', linestyle='--', linewidth=0.5)
    ax_st.set_title(f'Given vs. learned priors (episode length)')
    ax_st.set_xlabel('Episodes')
    ax_st.set_ylabel('Steps until goal')

    for label, color, linestyle, results_file in results:
        with open(results_file, 'rb') as f:
            tmlns = pickle.load(f)
        plot_training_history(tmlns, ax=ax_st, label=label, color=color, linestyle=linestyle, alpha=0.4, smoothing=5)

    ax_st.legend(loc='upper right', framealpha=0.6)
    ax_st.set_xlim((0, 150))
    ax_st.set_xticks([0, 25, 50, 75, 100, 125, 150])
    ax_st.set_ylim((0, 1000))
    fig_st.tight_layout()
    fig_st.subplots_adjust(left=0.1, bottom=0.176, top=0.907, right=0.975)
    # plt.show()
    fig_st.savefig('./figures/goal_vs_rewards_episode_length.pdf')
    plt.close()


def phase_portraits_goal_vs_rewards():
    models = [
        ('Given prior, H=10', './simulation_results/model_given_prior_H10.pt', './settings_given_prior_H10.json'),
        ('Given prior, H=15', './simulation_results/model_given_prior_H15.pt', './settings_given_prior_H15.json'),
        ('Learned prior, H=5', './simulation_results/model_learned_prior_H5.pt', './settings_learned_prior_H5.json'),
    ]

    fig = plt.figure(figsize=(8, 2.4))
    axes = []
    for i, (model_name, model_file, settings_file) in enumerate(models):
        axis = fig.add_subplot(1, 3, i + 1)

        args = Args(settings=settings_file,
                    load_existing=True,
                    save_dirpath='',
                    model_load_filepath=model_file)
        settings = args_to_simulation_settings(args)
        settings['simulation']['episode_callbacks'] = [lambda agent, episode_history, observations_mapper, **kwargs: _plot_phase_portrait(fig, axis, agent, episode_history, observations_mapper, label_cbar=i == 2, show_t_goal=True)]
        settings['simulation']['episodes'] = 1
        settings['simulation']['train_parameters'] = False

        # Run an episode and plot the phase portrait onto the axis
        run_training(
            settings['agent'],
            **settings['simulation'],
        )

        axis.set_title(model_name)
        axes.append(axis)
    axes[1].set_ylabel('')
    axes[2].set_ylabel('')
    fig.tight_layout()
    fig.subplots_adjust(left=0.063, bottom=0.058, top=1.0, right=0.943)
    # plt.show()
    plt.savefig('./figures/goal_vs_rewards_phase_portraits.pdf')
    plt.close()


def clean_vs_noise():
    results = [
        ('Learned prior, H=5', (0.3, 0.5, 1.0), '-', './simulation_results/results_learned_prior_H5.pickle'),
        ('Learned prior, H=5, with noise', (0.8, 0.3, 0.3), '--', './simulation_results/results_learned_prior_H5_noise.pickle')
    ]

    fig = plt.figure(1, figsize=(8, 2.1))
    ax_st = fig.add_subplot(1, 2, 1)
    ax_fe = fig.add_subplot(1, 2, 2)

    ax_st.grid(linewidth=0.4, alpha=0.5)
    ax_st.axhline(200, color='k', linestyle='--', linewidth=0.5)
    ax_st.set_title(f'Effect of noise on episode length')
    ax_st.set_xlabel('Episodes')
    ax_st.set_ylabel('Steps until goal')

    ax_fe.grid(linewidth=0.4, alpha=0.5)
    ax_fe.set_title(f'Effect of noise on model free energy')
    ax_fe.set_xlabel('Episodes')
    ax_fe.set_ylabel('Cumulative free energy')

    for label, color, linestyle, results_file in results:
        with open(results_file, 'rb') as f:
            tmlns = pickle.load(f)
        plot_training_history(tmlns, ax=ax_st, label=label, color=color, linestyle=linestyle, smoothing=0)
        plot_training_free_energy(tmlns, ax=ax_fe, label=label, color=color, linestyle=linestyle, smoothing=0)

    ax_st.legend(framealpha=0.4)
    ax_fe.legend(framealpha=0.4)
    ax_st.set_xlim((0, 150))
    ax_st.set_xticks([0, 25, 50, 75, 100, 125, 150])
    ax_st.set_ylim((0, 1000))
    ax_fe.set_xlim((0, 150))
    ax_fe.set_xticks([0, 25, 50, 75, 100, 125, 150])
    ax_fe.set_ylim((-200, 3000))

    fig.tight_layout()
    fig.subplots_adjust(left=0.075, bottom=0.25, top=0.88, right=0.98, wspace=0.3)
    # plt.show()
    fig.savefig('./figures/clean_vs_noise.pdf')
    plt.close()


def observations_and_policy_noise():
    model_file = './simulation_results/model_learned_prior_H5_noise.pt'
    settings_file = 'settings_learned_prior_H5_noise.json'

    fig = plt.figure(figsize=(7, 6))
    gs = fig.add_gridspec(7, 1)
    ax_obs = fig.add_subplot(gs[:3, 0])
    ax_lat1 = fig.add_subplot(gs[3:5, 0])
    ax_lat2 = fig.add_subplot(gs[5:7, 0])

    def make_plot(agent: ActiveInferenceCapsule, episode_history: Timeline, **kwargs):
        merged_history = episode_history.merge(agent.logged_history)

        # 1) Plot actions and states
        _plot_observations_actions(ax_obs, agent, merged_history)
        ax_obs.set_title(f'Observations and policy. Learned prior, H=5, noise std={settings["simulation"]["observation_noise_std"]}')
        _plot_latent_prediction(ax_lat1, 0, merged_history)
        _plot_latent_prediction(ax_lat2, 1, merged_history)

    args = Args(settings=settings_file,
                load_existing=True,
                save_dirpath='',
                model_load_filepath=model_file)
    settings = args_to_simulation_settings(args)
    settings['simulation']['episode_callbacks'] = [make_plot]
    settings['simulation']['episodes'] = 1
    settings['simulation']['train_parameters'] = False

    # Run an episode and plot the phase portrait onto the axis
    stats = run_training(
        settings['agent'],
        **settings['simulation'],
    )

    ax_obs.set_xlim((0, 1.5 * stats.select_features(['steps_per_episode'])[1][-1]))
    ax_obs.set_xlabel('Steps')
    ax_lat1.set_xlim((0, 120))
    ax_lat2.set_xlim((0, 120))
    fig.tight_layout()
    # plt.show()
    plt.savefig('./figures/clean_vs_noise_observations_and_policy.pdf')
    plt.close()


def ablation_study():
    results = [
        ('Full model', (152 / 255, 57 / 255, 253 / 255), '-', './simulation_results/results_learned_prior_H5.pickle'),
        ('Only intrinsic term', (203 / 255, 161 / 255, 39 / 255), (0, (5, 1)), './simulation_results/results_learned_prior_H5_only_intrinsic.pickle'),
        ('Only extrinsic term', (244 / 255, 24 / 255, 70 / 255), 'dotted', './simulation_results/results_learned_prior_H5_only_extrinsic.pickle'),
        ('Only extrinsic term, 25-trial hot start', (80 / 255, 181 / 255, 255 / 255), '-.', './simulation_results/results_learned_prior_H5_only_extrinsic_hotstart25.pickle'),
    ]

    fig_st = plt.figure(1, figsize=(6, 2.5))
    ax_st = fig_st.gca()
    ax_st.grid(linewidth=0.4, alpha=0.5)
    ax_st.axhline(200, color='k', linestyle='--', linewidth=0.5)
    ax_st.set_title(f'Ablation study, learned prior, H=5')
    ax_st.set_xlabel('Episodes')
    ax_st.set_ylabel('Steps until goal')

    for i, (label, color, linestyle, results_file) in enumerate(results):
        with open(results_file, 'rb') as f:
            timelines = pickle.load(f)
        if 'hotstart' in results_file:
            timelines_converge = []
            timelines_diverge = []
            post_hot_start_times = [t for t in timelines[0].times if t >= 25]
            for timeline in timelines:
                if timeline.select_features('steps_per_episode')[1][-1] > 300:
                    timeline = timeline.resample(post_hot_start_times, keep_outside_times=False)
                    timeline.log(24, 'steps_per_episode', 135.0)
                    timelines_diverge.append(timeline)
                else:
                    timelines_converge.append(timeline)
            plot_training_history(timelines_diverge, ax=ax_st, label=label, color=color, linestyle=linestyle, alpha=0.4, smoothing=0)
            plot_training_history(timelines_converge, ax=ax_st, label=None, color=color, linestyle=linestyle, alpha=0.4, smoothing=5)
        else:
            plot_training_history(timelines, ax=ax_st, label=label, color=color, linestyle=linestyle, alpha=0.4, smoothing=5)

    ax_st.legend(loc='upper right', bbox_to_anchor=(1.0, 0.92), framealpha=0.6)
    ax_st.set_xlim((0, 150))
    ax_st.set_ylim((0, 1050))
    ax_st.set_xticks([0, 25, 50, 75, 100, 125, 150])
    fig_st.tight_layout()
    fig_st.subplots_adjust(left=0.1, bottom=0.176, top=0.907, right=0.975)
    # plt.show()
    fig_st.savefig('./figures/ablation_study.pdf')
    plt.close()


if __name__ == '__main__':
    goal_vs_rewards()
    phase_portraits_goal_vs_rewards()
    clean_vs_noise()
    observations_and_policy_noise()
    ablation_study()
