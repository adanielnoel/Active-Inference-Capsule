from typing import Type

import torch
import torch.nn as nn
from torch.distributions.distribution import Distribution
import torch.distributions as distr
import torch.optim as optim
import torch.autograd.profiler as profiler


class PredictorGRU(nn.Module):
    """
    Predictor model for latent state sequences under a policy (i.e., a sequence of control actions)

    Args:
        latent_dim: number of dimensions of the latent space
        policy_dim: number of dimensions of the policy space
        dynamic_dim: number of dimensions of the dynamic states (i.e., hidden states of the `RNN`)
        num_rnn_layers: number of RNN layers
        post_dist: type of posterior distribution over hidden states. Should be equal to the one for q_x_y. Defaults to `Normal`.
    """

    def __init__(self, latent_dim: int, policy_dim: int, dynamic_dim: int, num_rnn_layers: int, post_dist: Type[Distribution] = distr.Normal):
        super(PredictorGRU, self).__init__()
        self.post_dist = post_dist
        self.latent_dim = latent_dim
        self.policy_dim = policy_dim
        self.dynamic_dim = dynamic_dim
        self.num_rnn_layers = num_rnn_layers

        # Neural networks:
        self.gru = nn.GRU(latent_dim + policy_dim, dynamic_dim, num_layers=num_rnn_layers, dropout=0.05 if num_rnn_layers > 1 else 0)
        self.mu = nn.Linear(dynamic_dim, latent_dim)
        self.scale = nn.Linear(dynamic_dim, latent_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x: torch.Tensor, policy: torch.Tensor, dyn: torch.Tensor = None):
        """
        Predicts the latent state one time-step ahead
        :param x: [sequence_length, 1, latent_dim]
        :param policy: [sequence_length, 1, policy_dim]
        :param dyn: An initial dynamic state (e.g., from the previous call). Shape: [num_rnn_layers, 1, dynamic_dim]
        :return: the mean and variance of the density of the predicted latent, and the dynamic state
        """
        pred, dyn = self.gru(torch.cat((x, policy), dim=2), dyn)
        return self.mu(pred) + x, nn.functional.softplus(self.scale(pred)) + 1e-6, dyn

    def predict(self, policy: torch.Tensor, dyn, first_latent):
        """
        Predicts the latent state several time-steps ahead under the provided policy

        #TODO: fix doc after changes
        The following diagram shows the inputs to each step:

            | INPUTS            | OUTPUTS
        i=0 | dyn               | q_x_ya_pred1
        i=1 | q_x_ya_pred1, a_1 | q_x_ya_pred2
        i=2 | q_x_ya_pred2, a_2 | q_x_ya_pred3
        i=3 | q_x_ya_pred3, a_3 | q_x_ya_pred4
        ... | ...               | ...

        :param policy: sequence of actions into the future [prediction_length, batch_size, policy_dim]
        :param dyn: An initial dynamic state (e.g., from the previous call) [num_rnn_layers, batch_size, dynamic_dim]
        :return: the mean and variance of the predicted latents
        """

        steps, n_policies = policy.shape[:2]
        next_mu = []
        next_var = []

        if dyn.shape[1] == 1 != n_policies:  # If dyn state is not batched, use the same for each batch
            dyn = dyn.expand((dyn.shape[0], n_policies, dyn.shape[2]))
        if first_latent.shape[1] == 1 != n_policies:  # If first_latent state is not batched, use the same for each batch
            first_latent = first_latent.expand((1, n_policies, first_latent.shape[2]))

        last_dyn = dyn
        for i in range(steps):
            _mu, _var, _dyn = self(first_latent if i == 0 else next_mu[i - 1], policy[[i]], last_dyn)
            next_mu.append(_mu)
            next_var.append(_var)
            last_dyn = _dyn.clone().detach()  # do not backprop through the hidden state

        return torch.cat(next_mu), torch.cat(next_var)

    def perceive_policy_outcome(self, px_y: distr.Distribution, policy: torch.Tensor, dyn: torch.Tensor, first_latent: torch.Tensor):
        with profiler.record_function("Retrospection"):
            prior_latents = torch.cat((first_latent, px_y.mean[:-1].unsqueeze(1)), dim=0)  # A sequence of latents that are prior to performing the actions
            pred_latent_locs, pred_latent_scales, dyn = self(prior_latents, policy.unsqueeze(1), dyn)
            pred_latent_seq_density = distr.Normal(pred_latent_locs.squeeze(1), pred_latent_scales.squeeze(1))
        return dyn, pred_latent_seq_density

    def learn_policy_outcome(self, px_y: distr.Distribution, policy: torch.Tensor, dyn: torch.Tensor, first_latent: torch.Tensor):
        """
        Performs a stochastic gradient descent step on the predictor model.

        This trains the predictor model on the new policy-observation sequence. The loss is the
        divergence between the latent distributions delivered by the observation model and the latent
        distributions predicted by the predictor model one step ahead from all previous observations.

        Inputs
            x      :      [y_1, y_2, y_3, y_4, y_5, ...]
            policy : [a_0, a_1, a_2, a_3, ...]

            px_y.shape = [sequence_length, latent_dim]
            policy.shape = [sequence_length, policy_dim]
            dyn.shape = [num_rnn_layers, 1, dynamic_dim]
            first_latent.shape = [1, 1, latent_dim]

        Note: y_0 is the one from the previous call, assuming there has not been any time-gap.
              For instance, y_5 of this retrospect call will be y_1 of the next one.
        """

        dyn, pred_latent_seq_density = self.perceive_policy_outcome(px_y, policy, dyn, first_latent)

        with profiler.record_function("Transition backprop"):
            self.optimizer.zero_grad()
            loss = distr.kl_divergence(px_y, pred_latent_seq_density).sum()  # Total divergence is sum of divergences for each latent component at every time-step
            loss.backward()
            self.optimizer.step()

        return dyn
