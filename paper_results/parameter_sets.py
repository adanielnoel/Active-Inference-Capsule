"""
These are the distinctive settings of every agent in the paper.
They correspond to parameters in training.py or batch_training.py
This is not a script but a file for reference. The parameters need to be entered by hand.
"""

name = 'results_given_prior_H15'
_learn_prior_model = False
_include_cart_velocity = True
_observation_noise_std = None
_time_compression = 6
_planning_horizon = 15
n_policy_samples = 1500,
policy_iterations = 2,
n_policy_candidates = 70,
action_window = 2
# > Batch training time ~30min for 15 agents in 8 parallel processes


name = 'results_given_prior_H10'
_learn_prior_model = False
_include_cart_velocity = True
_observation_noise_std = None
_time_compression = 6
_planning_horizon = 10
n_policy_samples = 700,
policy_iterations = 2,
n_policy_candidates = 70,
action_window = 2
# > Batch training time ~25min for 15 agents in 8 parallel processes


name = 'results_random'
num_episodes = 150
_learn_prior_model = False
_include_cart_velocity = True
_observation_noise_std = None
_time_compression = 6
_planning_horizon = 3
policy_dim = 1,
n_policy_samples = 1,  # this makes it take random policies
policy_iterations = 1,  # this makes it take random policies
n_policy_candidates = 1,  # this makes it take random policies
action_window = 2
# > Batch training time ~7min for 15 agents in 8 parallel processes


name = 'results_learned_prior_H10'
_learn_prior_model = True
prior_model = BiasedModelBellman(observation_dim=2 if _include_cart_velocity else 1, learning_rate=0.1, iterate_train=15, discount_factor=0.995)
_include_cart_velocity = True
_observation_noise_std = None
_time_compression = 6
_planning_horizon = 10
n_policy_samples = 700,
policy_iterations = 2,
n_policy_candidates = 70,
action_window = 2
# > Batch training time ~10min for 15 agents in 8 parallel processes


name = 'results_learned_prior_H5'
_learn_prior_model = True
prior_model = BiasedModelBellman(observation_dim=2 if _include_cart_velocity else 1, learning_rate=0.1, iterate_train=15, discount_factor=0.995)
_include_cart_velocity = True
_observation_noise_std = None
_time_compression = 6
_planning_horizon = 5
n_policy_samples = 700,
policy_iterations = 2,
n_policy_candidates = 70,
action_window = 2
# > Batch training time ~5min for 15 agents in 8 parallel processes


name = 'results_learned_prior_H5_noise'
# <same as results_learned_prior_H5>
_observation_noise_std = 0.1
# > Batch training time ~5min for 15 agents in 8 parallel processes


name = 'results_learned_prior_H5_only_intrinsic'
# <same as results_learned_prior_H5>
use_kl_extrinsic = False
# > Batch training time ~7min for 15 agents in 8 parallel processes


name = 'results_learned_prior_H5_only_extrinsic'
# <same as results_learned_prior_H5>
use_kl_intrinsic = False
# > Batch training time ~60min for 15 agents in 8 parallel processes


name = 'results_learned_prior_H5_only_extrinsic_hotstart25'
# <same as results_learned_prior_H5>
use_kl_intrinsic = False
hot_start_episodes = 25
# > Batch training time ~20min for 15 agents in 8 parallel processes
