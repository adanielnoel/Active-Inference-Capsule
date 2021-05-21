import pickle

import matplotlib.pyplot as plt

from mountain_car.plotting import plot_training_history, plot_training_free_energy

results = [
    # (label, color, line style, results file)
    ('Random agent', (190 / 255, 190 / 255, 190 / 255), 'dotted', '../paper_results/results_random.pickle'),
    ('Given prior, H=15', (255 / 255, 35 / 255, 100 / 255), '--', '../paper_results/results_given_prior_H15.pickle'),
    ('Given prior, H=10', (203 / 255, 161 / 255, 39 / 255), '-.', '../paper_results/results_given_prior_H10.pickle'),
    ('Learned prior, H=10', (100 / 255, 180 / 255, 255 / 255), (0, (3, 1, 1, 1, 1, 1)), '../paper_results/results_learned_prior_H10.pickle'),
    ('Learned prior, H=5', (210 / 255, 100 / 255, 255 / 255), '-', '../paper_results/results_learned_prior_H5.pickle'),
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


for label, color, linestyle, results_file in results:
    with open(results_file, 'rb') as f:
        tmlns = pickle.load(f)
    plot_training_history(tmlns, ax=fig_st.gca(), label=label, color=color, linestyle=linestyle, smoothing=0)
    plot_training_free_energy(tmlns, ax=fig_fe.gca(), label=label, color=color, linestyle=linestyle)

fig_fe.gca().set_ylim((-500, 3000))
fig_st.gca().legend()
fig_fe.gca().legend()
fig_st.tight_layout()
fig_fe.tight_layout()
plt.show()
