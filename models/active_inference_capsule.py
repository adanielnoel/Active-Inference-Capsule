from typing import Union, Iterable

import torch
import torch.nn as nn
import torch.distributions as distr
import torch.autograd.profiler as profiler

from models.vae import VAE
from models.transition_model import PredictorGRU
from utils.timeline import Timeline


class ActiveInferenceCapsule(nn.Module):
    """
    Learns observation, transition and biased generative models, and
    generates policies with minimal Free Energy of the Expected Future [1].

    Use:
        Call step() at every time-step. The function returns the action to take next.
        If using a learned prior, call learn_biased_model() when the goal is reached.

    References:
        - [1] B. Millidge et al, “Whence the Expected Free Energy?,” 2020, doi: 10.1162/neco_a_01354.
    """

    def __init__(self,
                 vae: VAE,
                 biased_model,
                 policy_dim: int,
                 time_step_size: float,
                 action_window: int,
                 planning_horizon: int,
                 n_policy_samples: int,
                 policy_iterations: int,
                 n_policy_candidates: int,
                 disable_kl_intrinsic=False,    # Use for ablation studies
                 disable_kl_extrinsic=False):   # Use for ablation studies
        super(ActiveInferenceCapsule, self).__init__()
        self.max_predicted_log_prob = 0.0

        # internal models
        self.vae = vae                          # p(x|y) and p(y|x)
        self.transition_model = PredictorGRU(   # p(x|𝜋)
            latent_dim=vae.latent_dim,
            policy_dim=policy_dim,
            dynamic_dim=planning_horizon * vae.latent_dim * 2,  # Make large enough for representing trajectories (heuristic)
            num_rnn_layers=1)
        self.biased_model = biased_model        # Given or learnable prior

        # Policy settings
        self.planning_horizon = planning_horizon
        self.n_policy_samples = n_policy_samples
        self.policy_iterations = policy_iterations
        self.n_policy_candidates = n_policy_candidates
        self.action_window = action_window
        self.use_kl_intrinsic = not disable_kl_intrinsic
        self.use_kl_extrinsic = not disable_kl_extrinsic

        # Short-term memory
        self.policy = None
        self.policy_std = None      # For logging only
        self.next_FEEFs = None      # For logging only
        self.next_locs = None       # For logging only
        self.next_scales = None     # For logging only
        self.new_observations = []
        self.new_actions = []
        self.new_times = []

        # Long-term memory
        self.time_step_size = time_step_size
        self.logged_history = Timeline()

        self.reset_states()

    def reset_states(self):
        self.transition_model.reset_states()
        self.policy = torch.zeros((self.planning_horizon, self.policy_dim))
        self.policy_std = torch.zeros((self.planning_horizon, self.policy_dim))
        self.next_FEEFs = torch.zeros(self.planning_horizon)
        self.next_locs = torch.zeros((self.planning_horizon, self.latent_dim))
        self.next_scales = torch.zeros((self.planning_horizon, self.latent_dim))
        self.new_observations = []
        self.new_actions = []
        self.new_times = []
        self.logged_history = Timeline()

    @property
    def observation_dim(self):
        return self.vae.observation_dim

    @property
    def latent_dim(self):
        return self.vae.latent_dim

    @property
    def policy_dim(self):
        return self.transition_model.policy_dim

    @property
    def dynamic_dim(self):
        return self.transition_model.dynamic_dim

    def step(self, time, observation: Union[torch.Tensor, Iterable], action: Union[torch.Tensor, Iterable] = None) -> torch.Tensor:
        """
        observation.shape = [observation_dim]
        action.shape = [action_dim]
        """
        observation = observation if isinstance(observation, torch.Tensor) else torch.tensor(observation, dtype=torch.float32).view(-1)
        action = action if isinstance(action, torch.Tensor) or action is None else torch.tensor(action, dtype=torch.float32)
        self.logged_history.log(time, 'perceived_observations', observation)
        if action is not None:
            self.new_actions.append(action)
            self.new_observations.append(observation)
            self.new_times.append(time)
            self.logged_history.log(time, 'perceived_actions', action)
        else:
            # The agent is passively observing and not evaluating the outcome of any policy yet
            self.new_observations = []
            self.new_actions = []
            self.transition_model.reset_states() # Invalidate the previous dynamic states

        if len(self.new_actions) == self.action_window:
            # Evaluate outcomes of last policy and draw a new policy
            new_observations = torch.stack(self.new_observations)
            new_actions = torch.stack(self.new_actions)
            with profiler.record_function("Learn observations"):
                _, new_posterior = self.perceive_observations(new_observations)  # + learning step on self.vae if in training mode
            with profiler.record_function("Retrospect_actions"):
                expected_x_mean, expected_x_std = self.transition_model.learn_policy_outcome(new_posterior, new_actions)  # + learning step on self.transition_model if in training mode
            with profiler.record_function("Sample_policy"):
                self.policy, self.policy_std, self.next_FEEFs, self.next_locs, self.next_scales = self.sample_policy()
                pred_t = time + self.time_step_size * torch.arange(0, self.planning_horizon)

            # Log all relevant variables
            expected_posterior = distr.Normal(expected_x_mean, expected_x_std)
            expected_likelihood = self.vae.decode_density(expected_posterior.mean)
            # expected_likelihood = distr.Normal(self.vae.decode(expected_posterior.mean), 1.0)
            new_VFEs = (-expected_likelihood.log_prob(new_observations) + distr.kl_divergence(expected_posterior, new_posterior)).sum(1)
            self.logged_history.log(self.new_times, 'VFE', new_VFEs.detach())
            self.logged_history.log(self.new_times, 'perceived_locs', new_posterior.mean.detach())
            self.logged_history.log(self.new_times, 'perceived_stds', new_posterior.stddev.detach())
            self.logged_history.log(self.new_times, 'filtered_observations_locs', expected_likelihood.mean.detach())
            self.logged_history.log(self.new_times, 'filtered_observations_stds', expected_likelihood.stddev.detach())
            self.logged_history.log(self.new_times, 'observations', self.new_observations)
            prediction = Timeline()
            prediction.log(pred_t, 'policy', self.policy)
            prediction.log(pred_t, 'policy_std', self.policy_std)
            prediction.log(pred_t, 'FEEF', self.next_FEEFs)
            prediction.log(pred_t, 'pred_locs', self.next_locs)
            prediction.log(pred_t, 'pred_stds', self.next_scales)
            self.logged_history.log(time, 'predictions', prediction)

            # Reset action window percepts
            self.new_observations = []
            self.new_actions = []
            self.new_times = []

        action = self.policy[len(self.new_actions)]
        action_std = self.policy_std[len(self.new_actions)]
        self.logged_history.log(time, 'actions_loc', action)
        self.logged_history.log(time, 'actions_std', action_std)
        self.logged_history.log(time, 'expected_locs', self.next_locs[len(self.new_actions)])
        self.logged_history.log(time, 'expected_stds', self.next_scales[len(self.new_actions)])
        self.logged_history.log(time, 'expected_FEEF', self.next_FEEFs[len(self.new_actions)])
        return action

    def kl_extrinsic(self, y):
        if isinstance(self.biased_model, distr.Normal):
            kl_extrinsic = distr.kl_divergence(distr.Normal(y, 1.0), self.biased_model).sum(dim=-1)  # Sum over components, keep time-steps and batches
        else:
            # Surrogate for non-random modelled priors
            kl_extrinsic = (1.0 - self.biased_model(y)).sum(dim=-1)  # Sum over components, keep time-steps and batches
        return kl_extrinsic

    def _forward_policies(self, policies: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """Forward propagate a batch of policies in time and compute their FEEFs
        Note:
            policies.shape = [planning_horizon, n_policies, policy_dim]

        References:
        - [1] B. Millidge et al, “Whence the Expected Free Energy?,” 2020, doi: 10.1162/neco_a_01354.
        """
        with profiler.record_function("Policy propagation"):
            next_x_means, next_x_stds = self.transition_model.predict(policies)
            policy_posterior = distr.Normal(next_x_means, next_x_stds)
        with profiler.record_function("FEEF"):
            with profiler.record_function("Latent reconstruction"):
                next_likelihoods = self.vae.decode_density(next_x_means)
                next_posteriors = self.vae.infer_density(next_likelihoods.mean)

            # Compute both KL components
            kl_extrinsic = self.kl_extrinsic(next_likelihoods.mean)
            kl_intrinsic = distr.kl_divergence(next_posteriors, policy_posterior).sum(dim=2)  # Sum over components, keep time-steps and batches
            # Disable components if doing an ablation study
            kl_extrinsic = kl_extrinsic if self.use_kl_extrinsic else torch.zeros_like(kl_extrinsic)
            kl_intrinsic = kl_intrinsic if self.use_kl_intrinsic else torch.zeros_like(kl_intrinsic)

            FEEFs = kl_extrinsic - kl_intrinsic
        return FEEFs, next_x_means, next_x_stds

    def sample_policy(self):
        """
        Implementation of the CEM algorithm.
        Similarly as done in [1], originally proposed in [2].
        References:
            [1] Tschantz, A., Millidge, B., Seth, A. K., & Buckley, C. L. (2020). Reinforcement Learning through Active Inference. ArXiv. http://arxiv.org/abs/2002.12636
            [2] Rubinstein, R. Y. (1997). Optimization of computer simulation models with rare events. European Journal of Operational Research, 99(1), 89–112. https://doi.org/10.1016/S0377-2217(96)00385-2
        """

        mean_best_policies = torch.zeros([self.planning_horizon, self.policy_dim])
        std_best_policies = torch.ones([self.planning_horizon, self.policy_dim])
        for i in range(self.policy_iterations):
            policy_distr = distr.Normal(mean_best_policies, std_best_policies)
            policies = policy_distr.sample([self.n_policy_samples, ]).transpose(0, 1)
            FEEFs, next_x_means, next_x_stds = self._forward_policies(policies.clamp(-1.0, 1.0))  # Clamp needed to prevent policy explosion, since higher magnitudes are unknown to the predictor and yield higher intrinsic value
            min_FEEF, min_FEEF_indices = FEEFs.sum(0).topk(self.n_policy_candidates, largest=False, sorted=False)  # sum over timesteps to get integrated FEEF for each policy, then pick the indices of the lowest
            mean_best_policies = policies[:, min_FEEF_indices].mean(1)
            std_best_policies = policies[:, min_FEEF_indices].std(1)

        # One last forward pass to gather the stats of the policy mean
        FEEFs, next_x_means, next_x_stds = self._forward_policies(mean_best_policies.unsqueeze(1))
        return mean_best_policies, std_best_policies, FEEFs.detach().squeeze(1), next_x_means.detach().squeeze(1), next_x_stds.detach().squeeze(1)

    def perceive_observations(self, y: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Returns the variational free energy for each observation and the encoded latents
        Trains the variational autoencoder on the new observations if in training mode
        """
        if self.training:
            with profiler.record_function("Learn VAE"):
                VFE, qx_y = self.vae.learn(y)
        else:
            with profiler.record_function("Forward VAE loss"):
                VFE, qx_y = self.vae.loss(y)

        # Return last states detached from graph to avoid spurious backpropagation in other functions
        qx_y.loc.detach_()
        qx_y.scale.detach_()
        return VFE.detach(), qx_y

    def learn_biased_model(self):
        if not isinstance(self.biased_model, distr.Distribution):
            _, po = self.logged_history.select_features('perceived_observations')
            self.biased_model.learn(torch.stack(po))
