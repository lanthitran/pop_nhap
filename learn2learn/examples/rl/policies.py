#!/usr/bin/env python3

import math

import cherry as ch
import torch
from torch import nn
from torch.distributions import Normal, Categorical

"""
This module implements different policy networks for reinforcement learning.
It provides three main policy classes: CaviaDiagNormalPolicy, DiagNormalPolicy, and CategoricalPolicy.
These policies are used to map states to actions in RL environments.
| Hung |
"""

# Small constant to prevent numerical instability in calculations  | Hung |
EPSILON = 1e-6


def linear_init(module):
    """
    Initializes linear layers using Xavier uniform initialization and zero bias.
    This helps with training stability and convergence.
    | Hung |
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()
    return module


class CaviaDiagNormalPolicy(nn.Module):
    """
    A policy network that uses a diagonal normal distribution for continuous action spaces.
    Includes context parameters for meta-learning capabilities.
    | Hung |
    """

    def __init__(self, input_size, output_size, hiddens=None, activation='relu', num_context_params=2, device='cpu'):
        """
        Initializes the CaviaDiagNormalPolicy.
        | Hung |
        """
        super(CaviaDiagNormalPolicy, self).__init__()
        self.device = device
        if hiddens is None:
            hiddens = [100, 100]
        if activation == 'relu':
            activation = nn.ReLU
        elif activation == 'tanh':
            activation = nn.Tanh
        # Build neural network layers with context parameters  | Hung |
        layers = [linear_init(nn.Linear(input_size+num_context_params, hiddens[0])), activation()]
        for i, o in zip(hiddens[:-1], hiddens[1:]):
            layers.append(linear_init(nn.Linear(i, o)))
            layers.append(activation())
        layers.append(linear_init(nn.Linear(hiddens[-1], output_size)))

        self.num_context_params = num_context_params
        # Initialize context parameters for meta-learning      | Hung |
        self.context_params = torch.zeros(self.num_context_params, requires_grad=True).to(self.device)

        self.mean = nn.Sequential(*layers).to(self.device)
        # Initialize standard deviation parameter     | Hung |
        self.sigma = nn.Parameter(torch.Tensor(output_size)).to(self.device)
        self.sigma.data.fill_(math.log(1))

    def density(self, state):
        """
        Computes the probability density function for the current state.
        | Hung |
        """
        state = state.to(self.device, non_blocking=True)
        # concatenate context parameters to input
        state = torch.cat((state, self.context_params.expand(state.shape[:-1] + self.context_params.shape)),
                          dim=len(state.shape) - 1)

        loc = self.mean(state)
        # Ensure minimum scale to prevent numerical issues      | Hung |
        scale = torch.exp(torch.clamp(self.sigma, min=math.log(EPSILON)))
        return Normal(loc=loc, scale=scale)

    def log_prob(self, state, action):
        """
        Computes the log probability of taking an action in a given state.
        | Hung |
        """
        density = self.density(state)
        return density.log_prob(action).mean(dim=1, keepdim=True)

    def forward(self, state):
        """
        Samples an action from the policy distribution.
        | Hung |
        """
        density = self.density(state)
        action = density.sample()
        return action

    def reset_context(self):
        """
        Resets the context parameters to zero.
        | Hung |
        """
        self.context_params[:] = 0  # torch.zeros(self.num_context_params, requires_grad=True).to(self.device)


class DiagNormalPolicy(nn.Module):
    """
    A simpler version of the policy network using diagonal normal distribution,
    without context parameters for meta-learning.
    | Hung |
    """

    def __init__(self, input_size, output_size, hiddens=None, activation='relu', device='cpu'):
        super(DiagNormalPolicy, self).__init__()
        self.device = device
        if hiddens is None:
            hiddens = [100, 100]
        if activation == 'relu':
            activation = nn.ReLU
        elif activation == 'tanh':
            activation = nn.Tanh
        # Build neural network layers    | Hung |
        layers = [linear_init(nn.Linear(input_size, hiddens[0])), activation()]
        for i, o in zip(hiddens[:-1], hiddens[1:]):
            layers.append(linear_init(nn.Linear(i, o)))
            layers.append(activation())
        layers.append(linear_init(nn.Linear(hiddens[-1], output_size)))
        self.mean = nn.Sequential(*layers)
        # Initialize standard deviation parameter      | Hung |
        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(1))

    def density(self, state):
        """
        Computes the probability density function for the current state.
        | Hung |
        """
        state = state.to(self.device, non_blocking=True)
        loc = self.mean(state)
        scale = torch.exp(torch.clamp(self.sigma, min=math.log(EPSILON)))
        return Normal(loc=loc, scale=scale)

    def log_prob(self, state, action):
        """
        Computes the log probability of taking an action in a given state.
        | Hung |
        """
        density = self.density(state)
        return density.log_prob(action).mean(dim=1, keepdim=True)

    def forward(self, state):
        """
        Samples an action from the policy distribution.
        | Hung |
        """
        density = self.density(state)
        action = density.sample()
        return action


class CategoricalPolicy(nn.Module):
    """
    A policy network for discrete action spaces using categorical distribution.
    | Hung |
    """

    def __init__(self, input_size, output_size, hiddens=None):
        super(CategoricalPolicy, self).__init__()
        if hiddens is None:
            hiddens = [100, 100]
        # Build neural network layers     | Hung |
        layers = [linear_init(nn.Linear(input_size, hiddens[0])), nn.ReLU()]
        for i, o in zip(hiddens[:-1], hiddens[1:]):
            layers.append(linear_init(nn.Linear(i, o)))
            layers.append(nn.ReLU())
        layers.append(linear_init(nn.Linear(hiddens[-1], output_size)))
        self.mean = nn.Sequential(*layers)
        self.input_size = input_size

    def forward(self, state):
        """
        Computes action probabilities and samples an action.
        Returns both the action and additional information about the distribution.
        | Hung |
        """
        state = ch.onehot(state, dim=self.input_size)
        loc = self.mean(state)
        density = Categorical(logits=loc)
        action = density.sample()
        log_prob = density.log_prob(action).mean().view(-1, 1).detach()
        return action, {'density': density, 'log_prob': log_prob}
