from typing import Union, Iterable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distr
import torch.autograd.profiler as profiler

from models.vae import VAE
from models.transition_model import PredictorGRU
from utils.timeline import Timeline


class ActiveInferenceCapsule(nn.Module):
    """
    An active inference (AIF) capsule is the main building block for active inference agents.
    It performs learning of the observation, transition and biased generative model, and
    deliberates over policies to select the one with the lowest Free Energy of the Expected Future.

    Use:
        An AIF capsule keeps track of the last known latent state and dynamic states, and projects
        predictions from them. Therefore, it is important that all observations and executed actions
        are processed, to keep account of the correct states. This can be done by passing observation
        and action sequences to `learn_policy_outcome`. This will both train the predictor on the
        sequence (one SGD step) and save the last latent and dynamic states.

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
                 disable_kl_intrinsic=False,
                 disable_kl_extrinsic=False):
        super(ActiveInferenceCapsule, self).__init__()
        self.max_predicted_log_prob = 0.0

        # internal models
        self.vae = vae
        self.biased_model = biased_model
        self.transition_model = PredictorGRU(latent_dim=vae.latent_dim,
                                             policy_dim=policy_dim,
                                             dynamic_dim=planning_horizon * vae.latent_dim * 1,  # Make large enough for representing trajectories (heuristic)
                                             num_rnn_layers=1,
                                             post_dist=vae.qx_y)

        # Policy settings
        self.planning_horizon = planning_horizon
        self.n_policy_samples = n_policy_samples
        self.policy_iterations = policy_iterations
        self.n_policy_candidates = n_policy_candidates
        self.action_window = action_window
        self.use_kl_intrinsic = not disable_kl_intrinsic
        self.use_kl_extrinsic = not disable_kl_extrinsic

        # Short-term memory
        self.dyn_state = None
        self.last_latent = None
        self.policy = None
        self.policy_std = None
        self.next_FEEFs = None
        self.next_locs = None
        self.next_scales = None
        self.new_observations = None
        self.new_actions = None  # [policy_dim]
        self.new_times = None

        # Long-term memory
        self.time_step_size = time_step_size
        self.logged_history: Timeline = None

        self.reset_states()

    def reset_states(self):
        self.dyn_state = torch.zeros((self.transition_model.num_rnn_layers, 1, self.transition_model.dynamic_dim))
        self.last_latent = torch.zeros((1, 1, self.latent_dim))
        # self.policy, _, _, _ = self.best_policy_of_n(1, n_best=1)
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
            self.last_latent = self.vae.infer(observation.view((1, 1, -1))).detach()

        if len(self.new_actions) == self.action_window:
            # Evaluate outcomes of last policy and draw a new policy
            new_observations = torch.stack(self.new_observations)
            new_actions = torch.stack(self.new_actions)
            with profiler.record_function("Learn observations"):
                new_VFEs, new_posterior_distr = self.perceive_observations(new_observations)  # + learning step on self.vae if in training mode
            with profiler.record_function("Retrospect_actions"):
                self.dyn_state = self.perceive_policy_outcome(new_posterior_distr, new_actions)  # + learning step on self.transition_model if in training mode
                self.last_latent = new_posterior_distr.mean[-1].reshape(1, 1, -1)
            with profiler.record_function("Sample_policy"):
                self.policy, self.policy_std, self.next_FEEFs, self.next_locs, self.next_scales = self.sample_policy()
                pred_t = time + self.time_step_size * torch.arange(0, self.planning_horizon)

            # Log all relevant variables
            self.logged_history.log(self.new_times, 'VFE', new_VFEs)
            self.logged_history.log(self.new_times, 'perceived_locs', new_posterior_distr.mean)
            self.logged_history.log(self.new_times, 'perceived_stds', new_posterior_distr.stddev)
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
        return action

    def kl_extrinsic(self, y):
        if isinstance(self.biased_model, distr.Distribution):
            kl_extrinsic = distr.kl_divergence(self.biased_model, distr.Normal(y, torch.tensor([1.0, 1.0]))).sum(dim=-1)  # Sum over components, keep time-steps and batches
        else:
            # Surrogate for non-random modelled priors
            kl_extrinsic = (1.0 - self.biased_model(y)).sum(dim=-1)  # Sum over components, keep time-steps and batches
        return kl_extrinsic

    def _forward_policies(self, policies: torch.Tensor, dyn: torch.Tensor, last_latent: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """Forward propagate a batch of policies in time and compute their FEEFs
        Note:
            policies.shape = [planning_horizon, n_policies, policy_dim]
            dyn.shape = [num_rnn_layers, 1 or n_policies, dynamic_dim]
            last_latent.shape = [1, 1 or n_policies, latent_dim]

        References:
        - [1] B. Millidge et al, “Whence the Expected Free Energy?,” 2020, doi: 10.1162/neco_a_01354.
        """
        with profiler.record_function("Policy propagation"):
            next_locs, next_scales = self.transition_model.predict(policies, dyn, last_latent)
        with profiler.record_function("FEEF"):
            with profiler.record_function("Latent reconstruction"):
                observations_density = self.vae.decode_density(next_locs)
                latent_reconstruction_density = self.vae.infer_density(observations_density.mean)

            # Compute both KL components
            kl_extrinsic = self.kl_extrinsic(observations_density.mean)
            kl_intrinsic = distr.kl_divergence(latent_reconstruction_density, distr.Normal(next_locs, next_scales)).sum(dim=2)  # Sum over components, keep time-steps and batches
            # Disable components if doing an ablation study
            kl_extrinsic = kl_extrinsic if self.use_kl_extrinsic else torch.zeros_like(kl_extrinsic)
            kl_intrinsic = kl_intrinsic if self.use_kl_intrinsic else torch.zeros_like(kl_intrinsic)

            FEEFs = kl_extrinsic - kl_intrinsic
        return FEEFs, next_locs, next_scales

    def sample_policy(self):
        """
        Uses the CEM algorithm.
        Similarly as done in [1], originally proposed in [2].
        References:
            [1] Tschantz, A., Millidge, B., Seth, A. K., & Buckley, C. L. (2020). Reinforcement Learning through Active Inference. ArXiv. http://arxiv.org/abs/2002.12636
            [2] Rubinstein, R. Y. (1997). Optimization of computer simulation models with rare events. European Journal of Operational Research, 99(1), 89–112. https://doi.org/10.1016/S0377-2217(96)00385-2
        """
        # Expand dyn_state and last_latent for batched policy prediction here to avoid doing it for every iteration in the transition model
        dyn_state = self.dyn_state.expand((self.transition_model.num_rnn_layers, self.n_policy_samples, self.transition_model.dynamic_dim))
        last_latent = self.last_latent.expand((1, self.n_policy_samples, self.latent_dim))

        mean_best_policies = torch.zeros([self.planning_horizon, self.policy_dim])
        std_best_policies = torch.ones([self.planning_horizon, self.policy_dim])
        for i in range(self.policy_iterations):
            policy_distr = distr.Normal(mean_best_policies, std_best_policies)
            policies = policy_distr.sample([self.n_policy_samples, ]).transpose(0, 1)
            FEEFs, next_locs, next_scales = self._forward_policies(policies.clamp(-1.0, 1.0), dyn_state, last_latent)  # Clamp needed to prevent policy explosion, since higher magnitudes are unknown to the predictor and yield higher intrinsic value
            min_FEEF, min_FEEF_indices = FEEFs.sum(0).topk(self.n_policy_candidates, largest=False, sorted=False)  # sum over timesteps to get integrated FEEF for each policy, then pick the indices of the lowest
            mean_best_policies = policies[:, min_FEEF_indices].mean(1)
            std_best_policies = policies[:, min_FEEF_indices].std(1)

        # One last forward pass to gather the stats of the policy mean
        FEEFs, next_locs, next_scales = self._forward_policies(mean_best_policies.unsqueeze(1), self.dyn_state, self.last_latent)
        return mean_best_policies, std_best_policies, FEEFs.detach().squeeze(1), next_locs.detach().squeeze(1), next_scales.detach().squeeze(1)

    def perceive_policy_outcome(self, px_y: distr.Normal, policy: torch.Tensor):
        """
        Performs a stochastic gradient descent step on the predictor model and updates the current dynamic and latent states.

        This trains the predictor model on the new policy-observation sequence. The loss is the
        divergence between the latent distributions delivered by the observation model and the latent
        distributions predicted by the predictor model one step ahead from all previous observations.

        Inputs
            px_y   :      [x_1, x_2, x_3, x_4, x_5, ...]
            policy : [a_0, a_1, a_2, a_3, ...]

            y.shape = [sequence_length, observation_dim]
            policy.shape = [sequence_length, policy_dim]

        Note: y_0 is the one from the previous call, assuming there has not been any time-gap.
              For instance, y_5 of this retrospect call will be y_1 of the next one.
        """
        if self.training:
            self.set_predictor_requires_grad(True)  # Make sure gradients are computed for the predictor model
            new_dyn_state = self.transition_model.learn_policy_outcome(px_y, policy, self.dyn_state, self.last_latent)
        else:
            new_dyn_state, _ = self.transition_model.perceive_policy_outcome(px_y, policy, self.dyn_state, self.last_latent)

        # Detach from graph to avoid spurious backpropagation in other functions
        return new_dyn_state.detach()

    def perceive_observations(self, y: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Trains the variational autoencoder on the new observations
        Returns the variational free energy for each observation and the encoded latents
        """
        VFE, qx_y = None, None
        if self.training:
            self.set_spatial_requires_grad(True)  # Make sure gradients are computed for the autoencoder weights
            with profiler.record_function("Learn VAE"):
                # Learns new observation until the variational free energy is below a threshold
                # Note: if this threshold is set too low it may damage previous learning. The purpose of multiple iterations
                #       is to quickly fix very bad reconstructions, not to force the VAE to overfit.
                while VFE is None or VFE.sum() > 1000:
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

    def set_spatial_requires_grad(self, flag):
        for parameter in self.vae.parameters():
            parameter.requires_grad = flag

    def set_predictor_requires_grad(self, flag):
        for parameter in self.transition_model.parameters():
            parameter.requires_grad = flag

    def set_biased_requires_grad(self, flag):
        for parameter in self.biased_model.parameters():
            parameter.requires_grad = flag

    def set_requires_grad(self, flag):
        self.set_spatial_requires_grad(flag)
        self.set_predictor_requires_grad(flag)
        self.set_biased_requires_grad(flag)
