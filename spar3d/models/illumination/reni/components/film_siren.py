"""FiLM Siren MLP as per https://marcoamonteiro.github.io/pi-GAN-website/."""

from typing import Optional

import numpy as np
import torch
from torch import nn


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.kaiming_normal_(
            m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
        )


def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(
                    -np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq
                )

    return init


def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class CustomMappingNetwork(nn.Module):
    def __init__(self, in_features, map_hidden_layers, map_hidden_dim, map_output_dim):
        super().__init__()

        self.network = []

        for _ in range(map_hidden_layers):
            self.network.append(nn.Linear(in_features, map_hidden_dim))
            self.network.append(nn.LeakyReLU(0.2, inplace=True))
            in_features = map_hidden_dim

        self.network.append(nn.Linear(map_hidden_dim, map_output_dim))

        self.network = nn.Sequential(*self.network)

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[
            ..., : torch.div(frequencies_offsets.shape[-1], 2, rounding_mode="floor")
        ]
        phase_shifts = frequencies_offsets[
            ..., torch.div(frequencies_offsets.shape[-1], 2, rounding_mode="floor") :
        ]

        return frequencies, phase_shifts


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        freq = freq.expand_as(x)
        phase_shift = phase_shift.expand_as(x)
        return torch.sin(freq * x + phase_shift)


class FiLMSiren(nn.Module):
    """FiLM Conditioned Siren network."""

    def __init__(
        self,
        in_dim: int,
        hidden_layers: int,
        hidden_features: int,
        mapping_network_in_dim: int,
        mapping_network_layers: int,
        mapping_network_features: int,
        out_dim: int,
        outermost_linear: bool = False,
        out_activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        assert self.in_dim > 0
        self.out_dim = out_dim if out_dim is not None else hidden_features
        self.hidden_layers = hidden_layers
        self.hidden_features = hidden_features
        self.mapping_network_in_dim = mapping_network_in_dim
        self.mapping_network_layers = mapping_network_layers
        self.mapping_network_features = mapping_network_features
        self.outermost_linear = outermost_linear
        self.out_activation = out_activation

        self.net = nn.ModuleList()

        self.net.append(FiLMLayer(self.in_dim, self.hidden_features))

        for _ in range(self.hidden_layers - 1):
            self.net.append(FiLMLayer(self.hidden_features, self.hidden_features))

        self.final_layer = None
        if self.outermost_linear:
            self.final_layer = nn.Linear(self.hidden_features, self.out_dim)
            self.final_layer.apply(frequency_init(25))
        else:
            final_layer = FiLMLayer(self.hidden_features, self.out_dim)
            self.net.append(final_layer)

        self.mapping_network = CustomMappingNetwork(
            in_features=self.mapping_network_in_dim,
            map_hidden_layers=self.mapping_network_layers,
            map_hidden_dim=self.mapping_network_features,
            map_output_dim=(len(self.net)) * self.hidden_features * 2,
        )

        self.net.apply(frequency_init(25))
        self.net[0].apply(first_layer_film_sine_init)

    def forward_with_frequencies_phase_shifts(self, x, frequencies, phase_shifts):
        """Get conditiional frequencies and phase shifts from mapping network."""
        frequencies = frequencies * 15 + 30

        for index, layer in enumerate(self.net):
            start = index * self.hidden_features
            end = (index + 1) * self.hidden_features
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        x = self.final_layer(x) if self.final_layer is not None else x
        output = self.out_activation(x) if self.out_activation is not None else x
        return output

    def forward(self, x, conditioning_input):
        """Forward pass."""
        frequencies, phase_shifts = self.mapping_network(conditioning_input)
        return self.forward_with_frequencies_phase_shifts(x, frequencies, phase_shifts)
