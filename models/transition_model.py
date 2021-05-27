import torch
import torch.nn as nn
import torch.distributions as distr
import torch.optim as optim
import torch.autograd.profiler as profiler


class PredictorGRU(nn.Module):
    """
    Predictor model for latent state sequences conditioned on a policy (i.e., a sequence of control actions)

    Args:
        latent_dim: number of dimensions of the latent space
        policy_dim: number of dimensions of the policy space
        dynamic_dim: number of dimensions of the dynamic states (i.e., hidden states of the `RNN`)
        num_rnn_layers: number of RNN layers
    """

    def __init__(self, latent_dim: int, policy_dim: int, dynamic_dim: int, num_rnn_layers: int):
        super(PredictorGRU, self).__init__()
        # Hyperparameters
        self.latent_dim = latent_dim
        self.policy_dim = policy_dim
        self.dynamic_dim = dynamic_dim
        self.num_rnn_layers = num_rnn_layers

        # Neural networks:
        self.gru = nn.GRU(input_size=latent_dim + policy_dim, hidden_size=dynamic_dim, num_layers=num_rnn_layers, dropout=0.05 if num_rnn_layers > 1 else 0)
        self.mean_net = nn.Linear(dynamic_dim, latent_dim)
        self.std_net = nn.Linear(dynamic_dim, latent_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        # States
        self.prev_dyn_state = None  # The dynamical states before the latest observation
        self.latest_latent = None     # The latent state corresponding to the latest observation
        self.reset_states()

    def reset_states(self):
        self.prev_dyn_state = torch.zeros((self.num_rnn_layers, 1, self.dynamic_dim))
        self.latest_latent = torch.zeros((1, 1, self.latent_dim))

    def forward(self, x: torch.Tensor, policy: torch.Tensor, dyn: torch.Tensor = None):
        """
        Predicts the latent state one time-step ahead
        :param x: [sequence_length, n_policies, latent_dim]
        :param policy: [sequence_length, n_policies, policy_dim]
        :param dyn: An initial dynamic state (e.g., from the previous call). Shape: [num_rnn_layers, n_policies, dynamic_dim]
        :return: the mean and variance of the density of the predicted latent, and the dynamic state
        """
        pred, dyn = self.gru(torch.cat((x, policy), dim=2), dyn)
        return self.mean_net(pred) + x, nn.functional.softplus(self.std_net(pred)) + 1e-6, dyn

    def predict(self, policy: torch.Tensor):
        """
        Predicts the latent state several time-steps ahead from the first_latent under the provided policy
        :param policy: [sequence_length, n_policies, policy_dim]
        :return: the mean and standard deviation of the predicted latents
        """
        steps, n_policies = policy.shape[:2]
        next_mu = []
        next_var = []

        # If policies are batched, use the same states for each batch
        if n_policies == 1:
            prev_dyn = self.prev_dyn_state
            first_latent = self.latest_latent
        else:
            prev_dyn = self.prev_dyn_state.expand((self.num_rnn_layers, n_policies, self.dynamic_dim))
            first_latent = self.latest_latent.expand((1, n_policies, self.latent_dim))

        for i in range(steps):
            latent_mean, latent_std, dyn = self(first_latent if i == 0 else next_mu[i - 1], policy[[i]], prev_dyn)
            next_mu.append(latent_mean)
            next_var.append(latent_std)
            prev_dyn = dyn.detach()

        return torch.cat(next_mu), torch.cat(next_var)

    def learn_policy_outcome(self, px_y: distr.Distribution, policy: torch.Tensor):
        """
        Updates the dynamic state and trains the neural networks if in training mode.
        :param px_y: a distribution over a sequence of latent states. [sequence_length, latent_dim]
        :param policy: a sequence of actions preceding the observations. [sequence_length, policy_dim]
        :param
        """

        with torch.no_grad():
            prior_latents = torch.cat((self.latest_latent, px_y.mean[:-1].unsqueeze(1)), dim=0)  # A sequence of latents that are prior to performing the actions

        with profiler.record_function("Retrospection"):
            pred_x_mean, pred_x_std, prev_dyn_state = self(prior_latents, policy.unsqueeze(1), self.prev_dyn_state)

        if self.training:
            with profiler.record_function("Transition backprop"):
                self.optimizer.zero_grad()
                pred_px_y = distr.Normal(pred_x_mean.squeeze(1), pred_x_std.squeeze(1))
                loss = distr.kl_divergence(pred_px_y, px_y).sum()  # Total divergence is sum of divergences for each latent component at every time-step
                loss.backward()
                self.optimizer.step()

        self.prev_dyn_state = prev_dyn_state.detach()
        self.latest_latent = px_y.mean[-1].reshape((1, 1, self.latent_dim)).detach()  # The result of the last action. Not included in the dynamic state yet

        return pred_x_mean.squeeze(1).detach(), pred_x_std.squeeze(1).detach()
