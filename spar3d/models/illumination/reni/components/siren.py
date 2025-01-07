"""Siren MLP https://www.vincentsitzmann.com/siren/"""

from typing import Optional

import numpy as np
import torch
from torch import nn


class SineLayer(nn.Module):
    """
    Sine layer for the SIREN network.
    """

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30.0
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class Siren(nn.Module):
    """Siren network.

    Args:
        in_dim: Input layer dimension
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        out_dim: Output layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
        out_activation: output activation function.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_layers: int,
        hidden_features: int,
        out_dim: Optional[int] = None,
        outermost_linear: bool = False,
        first_omega_0: float = 30,
        hidden_omega_0: float = 30,
        out_activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        assert self.in_dim > 0
        self.out_dim = out_dim if out_dim is not None else hidden_features
        self.outermost_linear = outermost_linear
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0
        self.hidden_layers = hidden_layers
        self.layer_width = hidden_features
        self.out_activation = out_activation

        self.net = []
        self.net.append(
            SineLayer(in_dim, hidden_features, is_first=True, omega_0=first_omega_0)
        )

        for _ in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_layer = nn.Linear(hidden_features, self.out_dim)

            with torch.no_grad():
                final_layer.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

            self.net.append(final_layer)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    self.out_dim,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if self.out_activation is not None:
            self.net.append(self.out_activation)

        self.net = nn.Sequential(*self.net)

    def forward(self, model_input):
        """Forward pass through the network"""
        output = self.net(model_input)
        return output
