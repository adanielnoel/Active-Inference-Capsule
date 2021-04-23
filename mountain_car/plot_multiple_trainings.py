import pickle

import matplotlib.pyplot as plt

from mountain_car.plotting import plot_training_history, plot_training_free_energy

# results = [
#     ('Bellman prior, 30 steps', (0.2, 0.4, 1.0), './experiments/batch_run/results_bellman30.pickle'),
#     ('Bellman prior, 30 steps, noise', (0.8, 0.3, 0.3), './experiments/batch_run/results_bellman30_noise.pickle')
# ]

results = [
    ('Normal prior, 90 steps', (0.2, 0.4, 1.0), './experiments/batch_run/results_normal90.pickle'),
    ('Normal prior, 60 steps', (0.8, 0.3, 0.3), './experiments/batch_run/results_normal60.pickle'),
    ('Bellman prior, 30 steps', (0.3, 0.6, 0.3), './experiments/batch_run/results_bellman30.pickle'),
    # ('Bellman prior, 60 steps', (0.3, 0.6, 0.3), './experiments/batch_run/results_bellman60.pickle')
]

fig_st = plt.figure(1, figsize=(6, 3))
fig_st.gca().grid(linewidth=0.4, alpha=0.5)
fig_st.gca().axhline(200, color='k', linestyle='--', linewidth=0.5)
fig_st.gca().set_title(f'Episode length statistics (15 agents)')
fig_st.gca().set_xlabel('Episodes')
fig_st.gca().set_ylabel('Steps until goal')

fig_fe = plt.figure(2, figsize=(6, 3))
fig_fe.gca().grid(linewidth=0.4, alpha=0.5)
fig_fe.gca().set_title(f'Free energy statistics (15 agents)')
fig_fe.gca().set_xlabel('Episodes')
fig_fe.gca().set_ylabel('Episode cumulative free energy')


# plt.close()
for label, color, results_file in results:
    with open(results_file, 'rb') as f:
        tmlns = pickle.load(f)
    plot_training_history(tmlns, figure=fig_st, label=label, color=color)
    plot_training_free_energy(tmlns, figure=fig_fe, label=label, color=color)

fig_st.gca().legend()
fig_fe.gca().legend()
fig_st.tight_layout()
fig_fe.tight_layout()
plt.show()
fig_fe.gca().set_ylim((-500, 2500))
fig_fe.savefig('./experiments/batch_run/goal_vs_rewards_free_energy.pdf')
fig_st.savefig('./experiments/batch_run/goal_vs_rewards_episode_length.pdf')
