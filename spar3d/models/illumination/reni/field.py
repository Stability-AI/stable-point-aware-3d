# Copyright 2023 The University of York. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by Mark Boss

"""RENI field"""

import contextlib
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch
from einops.layers.torch import Rearrange
from jaxtyping import Float
from torch import Tensor, nn

from spar3d.models.network import get_activation_module, trunc_exp
from spar3d.models.utils import BaseModule

from .components.film_siren import FiLMSiren
from .components.siren import Siren
from .components.transformer_decoder import Decoder
from .components.vn_layers import VNInvariant, VNLinear

# from nerfstudio.cameras.rays import RaySamples


def expected_sin(x_means: torch.Tensor, x_vars: torch.Tensor) -> torch.Tensor:
    """Computes the expected value of sin(y) where y ~ N(x_means, x_vars)

    Args:
        x_means: Mean values.
        x_vars: Variance of values.

    Returns:
        torch.Tensor: The expected value of sin.
    """

    return torch.exp(-0.5 * x_vars) * torch.sin(x_means)


class NeRFEncoding(torch.nn.Module):
    """Multi-scale sinousoidal encodings. Support ``integrated positional encodings`` if covariances are provided.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        min_freq_exp: float,
        max_freq_exp: float,
        include_input: bool = False,
        off_axis: bool = False,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.include_input = include_input

        self.off_axis = off_axis

        self.P = torch.tensor(
            [
                [0.8506508, 0, 0.5257311],
                [0.809017, 0.5, 0.309017],
                [0.5257311, 0.8506508, 0],
                [1, 0, 0],
                [0.809017, 0.5, -0.309017],
                [0.8506508, 0, -0.5257311],
                [0.309017, 0.809017, -0.5],
                [0, 0.5257311, -0.8506508],
                [0.5, 0.309017, -0.809017],
                [0, 1, 0],
                [-0.5257311, 0.8506508, 0],
                [-0.309017, 0.809017, -0.5],
                [0, 0.5257311, 0.8506508],
                [-0.309017, 0.809017, 0.5],
                [0.309017, 0.809017, 0.5],
                [0.5, 0.309017, 0.809017],
                [0.5, -0.309017, 0.809017],
                [0, 0, 1],
                [-0.5, 0.309017, 0.809017],
                [-0.809017, 0.5, 0.309017],
                [-0.809017, 0.5, -0.309017],
            ]
        ).T

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        out_dim = self.in_dim * self.num_frequencies * 2

        if self.off_axis:
            out_dim = self.P.shape[1] * self.num_frequencies * 2

        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def forward(
        self,
        in_tensor: Float[Tensor, "*b input_dim"],
        covs: Optional[Float[Tensor, "*b input_dim input_dim"]] = None,
    ) -> Float[Tensor, "*b output_dim"]:
        """Calculates NeRF encoding. If covariances are provided the encodings will be integrated as proposed
            in mip-NeRF.

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
            covs: Covariances of input points.
        Returns:
            Output values will be between -1 and 1
        """
        # TODO check scaling here but just comment it for now
        # in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        freqs = 2 ** torch.linspace(
            self.min_freq, self.max_freq, self.num_frequencies
        ).to(in_tensor.device)
        # freqs = 2 ** (
        #    torch.sin(torch.linspace(self.min_freq, torch.pi / 2.0, self.num_frequencies)) * self.max_freq
        # ).to(in_tensor.device)
        # freqs = 2 ** (
        #     torch.linspace(self.min_freq, 1.0, self.num_frequencies).to(in_tensor.device) ** 0.2 * self.max_freq
        # )

        if self.off_axis:
            scaled_inputs = (
                torch.matmul(in_tensor, self.P.to(in_tensor.device))[..., None] * freqs
            )
        else:
            scaled_inputs = (
                in_tensor[..., None] * freqs
            )  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(
            *scaled_inputs.shape[:-2], -1
        )  # [..., "input_dim" * "num_scales"]

        if covs is None:
            encoded_inputs = torch.sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1)
            )
        else:
            input_var = (
                torch.diagonal(covs, dim1=-2, dim2=-1)[..., :, None]
                * freqs[None, :] ** 2
            )
            input_var = input_var.reshape((*input_var.shape[:-2], -1))
            encoded_inputs = expected_sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1),
                torch.cat(2 * [input_var], dim=-1),
            )

        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)
        return encoded_inputs


class RENIField(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        """Configuration for model instantiation"""

        fixed_decoder: bool = False
        """Whether to fix the decoder weights"""
        equivariance: str = "SO2"
        """Type of equivariance to use: None, SO2, SO3"""
        axis_of_invariance: str = "y"
        """Which axis should SO2 equivariance be invariant to: x, y, z"""
        invariant_function: str = "GramMatrix"
        """Type of invariant function to use: GramMatrix, VN"""
        conditioning: str = "Concat"
        """Type of conditioning to use: FiLM, Concat, Attention"""
        positional_encoding: str = "NeRF"
        """Type of positional encoding to use. Currently only NeRF is supported"""
        encoded_input: str = "Directions"
        """Type of input to encode: None, Directions, Conditioning, Both"""
        latent_dim: int = 36
        """Dimensionality of latent code, N for a latent code size of (N x 3)"""
        hidden_layers: int = 3
        """Number of hidden layers"""
        hidden_features: int = 128
        """Number of hidden features"""
        mapping_layers: int = 3
        """Number of mapping layers"""
        mapping_features: int = 128
        """Number of mapping features"""
        num_attention_heads: int = 8
        """Number of attention heads"""
        num_attention_layers: int = 3
        """Number of attention layers"""
        out_features: int = 3  # RGB
        """Number of output features"""
        last_layer_linear: bool = False
        """Whether to use a linear layer as the last layer"""
        output_activation: str = "exp"
        """Activation function for output layer: sigmoid, tanh, relu, exp, None"""
        first_omega_0: float = 30.0
        """Omega_0 for first layer"""
        hidden_omega_0: float = 30.0
        """Omega_0 for hidden layers"""
        fixed_decoder: bool = False
        """Whether to fix the decoder weights"""
        old_implementation: bool = False
        """Whether to match implementation of old RENI, when using old checkpoints"""

    cfg: Config

    def configure(self):
        self.equivariance = self.cfg.equivariance
        self.conditioning = self.cfg.conditioning
        self.latent_dim = self.cfg.latent_dim
        self.hidden_layers = self.cfg.hidden_layers
        self.hidden_features = self.cfg.hidden_features
        self.mapping_layers = self.cfg.mapping_layers
        self.mapping_features = self.cfg.mapping_features
        self.out_features = self.cfg.out_features
        self.last_layer_linear = self.cfg.last_layer_linear
        self.output_activation = self.cfg.output_activation
        self.first_omega_0 = self.cfg.first_omega_0
        self.hidden_omega_0 = self.cfg.hidden_omega_0
        self.old_implementation = self.cfg.old_implementation
        self.axis_of_invariance = ["x", "y", "z"].index(self.cfg.axis_of_invariance)

        self.fixed_decoder = self.cfg.fixed_decoder
        if self.cfg.invariant_function == "GramMatrix":
            self.invariant_function = self.gram_matrix_invariance
        else:
            self.vn_proj_in = nn.Sequential(
                Rearrange("... c -> ... 1 c"),
                VNLinear(dim_in=1, dim_out=1, bias_epsilon=0),
            )
            dim_coor = 2 if self.cfg.equivariance == "SO2" else 3
            self.vn_invar = VNInvariant(dim=1, dim_coor=dim_coor)
            self.invariant_function = self.vn_invariance

        self.network = self.setup_network()

        if self.fixed_decoder:
            for param in self.network.parameters():
                param.requires_grad = False

            if self.cfg.invariant_function == "VN":
                for param in self.vn_proj_in.parameters():
                    param.requires_grad = False
                for param in self.vn_invar.parameters():
                    param.requires_grad = False

    @contextlib.contextmanager
    def hold_decoder_fixed(self):
        """Context manager to fix the decoder weights

        Example usage:
        ```
        with instance_of_RENIField.hold_decoder_fixed():
            # do stuff
        ```
        """
        prev_state_network = {
            name: p.requires_grad for name, p in self.network.named_parameters()
        }
        for param in self.network.parameters():
            param.requires_grad = False
        if self.cfg.invariant_function == "VN":
            prev_state_proj_in = {
                k: p.requires_grad for k, p in self.vn_proj_in.named_parameters()
            }
            prev_state_invar = {
                k: p.requires_grad for k, p in self.vn_invar.named_parameters()
            }
            for param in self.vn_proj_in.parameters():
                param.requires_grad = False
            for param in self.vn_invar.parameters():
                param.requires_grad = False

        prev_decoder_state = self.fixed_decoder
        self.fixed_decoder = True
        try:
            yield
        finally:
            # Restore the previous requires_grad state
            for name, param in self.network.named_parameters():
                param.requires_grad = prev_state_network[name]
            if self.cfg.invariant_function == "VN":
                for name, param in self.vn_proj_in.named_parameters():
                    param.requires_grad_(prev_state_proj_in[name])
                for name, param in self.vn_invar.named_parameters():
                    param.requires_grad_(prev_state_invar[name])
            self.fixed_decoder = prev_decoder_state

    def vn_invariance(
        self,
        Z: Float[Tensor, "B latent_dim 3"],
        D: Float[Tensor, "B num_rays 3"],
        equivariance: Literal["None", "SO2", "SO3"] = "SO2",
        axis_of_invariance: int = 1,
    ):
        """Generates a batched invariant representation from latent code Z and direction coordinates D.

        Args:
            Z: [B, latent_dim, 3] - Latent code.
            D: [B num_rays, 3] - Direction coordinates.
            equivariance: The type of equivariance to use. Options are 'None', 'SO2', 'SO3'.
            axis_of_invariance: The axis of rotation invariance. Should be 0 (x-axis), 1 (y-axis), or 2 (z-axis).

        Returns:
            Tuple[Tensor, Tensor]: directional_input, conditioning_input
        """
        assert 0 <= axis_of_invariance < 3, "axis_of_invariance should be 0, 1, or 2."
        other_axes = [i for i in range(3) if i != axis_of_invariance]

        B, latent_dim, _ = Z.shape
        _, num_rays, _ = D.shape

        if equivariance == "None":
            # get inner product between latent code and direction coordinates
            innerprod = torch.sum(
                Z.unsqueeze(1) * D.unsqueeze(2), dim=-1
            )  # [B, num_rays, latent_dim]
            z_input = (
                Z.flatten(start_dim=1).unsqueeze(1).expand(B, num_rays, latent_dim * 3)
            )  # [B, num_rays, latent_dim * 3]
            return innerprod, z_input

        if equivariance == "SO2":
            z_other = torch.stack(
                (Z[..., other_axes[0]], Z[..., other_axes[1]]), -1
            )  # [B, latent_dim, 2]
            d_other = torch.stack(
                (D[..., other_axes[0]], D[..., other_axes[1]]), -1
            ).unsqueeze(2)  # [B, num_rays, 1, 2]
            d_other = d_other.expand(
                B, num_rays, latent_dim, 2
            )  # [B, num_rays, latent_dim, 2]

            z_other_emb = self.vn_proj_in(z_other)  # [B, latent_dim, 1, 2]
            z_other_invar = self.vn_invar(z_other_emb)  # [B, latent_dim, 2]

            # Get invariant component of Z along the axis of invariance
            z_invar = Z[..., axis_of_invariance].unsqueeze(-1)  # [B, latent_dim, 1]

            # Innerproduct between projection of Z and D on the plane orthogonal to the axis of invariance.
            # This encodes the rotational information. This is rotation-equivariant to rotations of either Z
            # or D and is invariant to rotations of both Z and D.
            innerprod = (z_other.unsqueeze(1) * d_other).sum(
                dim=-1
            )  # [B, num_rays, latent_dim]

            # Compute norm along the axes orthogonal to the axis of invariance
            d_other_norm = torch.sqrt(
                D[..., other_axes[0]] ** 2 + D[..., other_axes[1]] ** 2
            ).unsqueeze(-1)  # [B num_rays, 1]

            # Get invariant component of D along the axis of invariance
            d_invar = D[..., axis_of_invariance].unsqueeze(-1)  # [B, num_rays, 1]

            directional_input = torch.cat(
                (innerprod, d_invar, d_other_norm), -1
            )  # [B, num_rays, latent_dim + 2]
            conditioning_input = (
                torch.cat((z_other_invar, z_invar), dim=-1)
                .flatten(1)
                .unsqueeze(1)
                .expand(B, num_rays, latent_dim * 3)
            )  # [B, num_rays, latent_dim * 3]

            return directional_input, conditioning_input

        if equivariance == "SO3":
            z = self.vn_proj_in(Z)  # [B, latent_dim, 1, 3]
            z_invar = self.vn_invar(z)  # [B, latent_dim, 3]
            conditioning_input = (
                z_invar.flatten(1).unsqueeze(1).expand(B, num_rays, latent_dim)
            )  # [B, num_rays, latent_dim * 3]
            # D [B, num_rays, 3] -> [B, num_rays, 1, 3]
            # Z [B, latent_dim, 3] -> [B, 1, latent_dim, 3]
            innerprod = torch.sum(
                Z.unsqueeze(1) * D.unsqueeze(2), dim=-1
            )  # [B, num_rays, latent_dim]
            return innerprod, conditioning_input

    def gram_matrix_invariance(
        self,
        Z: Float[Tensor, "B latent_dim 3"],
        D: Float[Tensor, "B num_rays 3"],
        equivariance: Literal["None", "SO2", "SO3"] = "SO2",
        axis_of_invariance: int = 1,
    ):
        """Generates an invariant representation from latent code Z and direction coordinates D.

        Args:
            Z (torch.Tensor): Latent code (B x latent_dim x 3)
            D (torch.Tensor): Direction coordinates (B x num_rays x 3)
            equivariance (str): Type of equivariance to use. Options are 'none', 'SO2', and 'SO3'
            axis_of_invariance (int): The axis of rotation invariance. Should be 0 (x-axis), 1 (y-axis), or 2 (z-axis).
                Default is 1 (y-axis).
        Returns:
            torch.Tensor: Invariant representation
        """
        assert 0 <= axis_of_invariance < 3, "axis_of_invariance should be 0, 1, or 2."
        other_axes = [i for i in range(3) if i != axis_of_invariance]

        B, latent_dim, _ = Z.shape
        _, num_rays, _ = D.shape

        if equivariance == "None":
            # get inner product between latent code and direction coordinates
            innerprod = torch.sum(
                Z.unsqueeze(1) * D.unsqueeze(2), dim=-1
            )  # [B, num_rays, latent_dim]
            z_input = (
                Z.flatten(start_dim=1).unsqueeze(1).expand(B, num_rays, latent_dim * 3)
            )  # [B, num_rays, latent_dim * 3]
            return innerprod, z_input

        if equivariance == "SO2":
            # Select components along axes orthogonal to the axis of invariance
            z_other = torch.stack(
                (Z[..., other_axes[0]], Z[..., other_axes[1]]), -1
            )  # [B, latent_dim, 2]
            d_other = torch.stack(
                (D[..., other_axes[0]], D[..., other_axes[1]]), -1
            ).unsqueeze(2)  # [B, num_rays, 1, 2]
            d_other = d_other.expand(
                B, num_rays, latent_dim, 2
            )  # size becomes [B, num_rays, latent_dim, 2]

            # Invariant representation of Z, gram matrix G=Z*Z' is size num_rays x latent_dim x latent_dim
            G = torch.bmm(z_other, torch.transpose(z_other, 1, 2))

            # Flatten G to be size B x latent_dim^2
            z_other_invar = G.flatten(start_dim=1)

            # Get invariant component of Z along the axis of invariance
            z_invar = Z[..., axis_of_invariance]  # [B, latent_dim]

            # Innerprod is size num_rays x latent_dim
            innerprod = (z_other.unsqueeze(1) * d_other).sum(
                dim=-1
            )  # [B, num_rays, latent_dim]

            # Compute norm along the axes orthogonal to the axis of invariance
            d_other_norm = torch.sqrt(
                D[..., other_axes[0]] ** 2 + D[..., other_axes[1]] ** 2
            ).unsqueeze(-1)  # [B, num_rays, 1]

            # Get invariant component of D along the axis of invariance
            d_invar = D[..., axis_of_invariance].unsqueeze(-1)  # [B, num_rays, 1]

            if not self.old_implementation:
                directional_input = torch.cat(
                    (innerprod, d_invar, d_other_norm), -1
                )  # [B, num_rays, latent_dim + 2]
                conditioning_input = (
                    torch.cat((z_other_invar, z_invar), -1)
                    .unsqueeze(1)
                    .expand(B, num_rays, latent_dim * 3)
                )  # [B, num_rays, latent_dim^2 + latent_dim]
            else:
                # this is matching the previous implementation of RENI, needed if using old checkpoints
                z_other_invar = z_other_invar.unsqueeze(1).expand(B, num_rays, -1)
                z_invar = z_invar.unsqueeze(1).expand(B, num_rays, -1)
                return torch.cat(
                    (innerprod, z_other_invar, d_other_norm, z_invar, d_invar), 1
                )

            return directional_input, conditioning_input

        if equivariance == "SO3":
            G = Z @ torch.transpose(Z, 1, 2)  # [B, latent_dim, latent_dim]
            innerprod = torch.sum(
                Z.unsqueeze(1) * D.unsqueeze(2), dim=-1
            )  # [B, num_rays, latent_dim]
            z_invar = (
                G.flatten(start_dim=1).unsqueeze(1).expand(B, num_rays, -1)
            )  # [B, num_rays, latent_dim^2]
            return innerprod, z_invar

    def setup_network(self):
        """Sets up the network architecture"""
        base_input_dims = {
            "VN": {
                "None": {
                    "direction": self.latent_dim,
                    "conditioning": self.latent_dim * 3,
                },
                "SO2": {
                    "direction": self.latent_dim + 2,
                    "conditioning": self.latent_dim * 3,
                },
                "SO3": {
                    "direction": self.latent_dim,
                    "conditioning": self.latent_dim * 3,
                },
            },
            "GramMatrix": {
                "None": {
                    "direction": self.latent_dim,
                    "conditioning": self.latent_dim * 3,
                },
                "SO2": {
                    "direction": self.latent_dim + 2,
                    "conditioning": self.latent_dim**2 + self.latent_dim,
                },
                "SO3": {
                    "direction": self.latent_dim,
                    "conditioning": self.latent_dim**2,
                },
            },
        }

        # Extract the necessary input dimensions
        input_types = ["direction", "conditioning"]
        input_dims = {
            key: base_input_dims[self.cfg.invariant_function][self.cfg.equivariance][
                key
            ]
            for key in input_types
        }

        # Helper function to create NeRF encoding
        def create_nerf_encoding(in_dim):
            return NeRFEncoding(
                in_dim=in_dim,
                num_frequencies=2,
                min_freq_exp=0.0,
                max_freq_exp=2.0,
                include_input=True,
            )

        # Dictionary-based encoding setup
        encoding_setup = {
            "None": [],
            "Conditioning": ["conditioning"],
            "Directions": ["direction"],
            "Both": ["direction", "conditioning"],
        }

        # Setting up the required encodings
        for input_type in encoding_setup.get(self.cfg.encoded_input, []):
            # create self.{input_type}_encoding and update input_dims
            setattr(
                self,
                f"{input_type}_encoding",
                create_nerf_encoding(input_dims[input_type]),
            )
            input_dims[input_type] = getattr(
                self, f"{input_type}_encoding"
            ).get_out_dim()

        output_activation = get_activation_module(self.cfg.output_activation)

        network = None
        if self.conditioning == "Concat":
            network = Siren(
                in_dim=input_dims["direction"] + input_dims["conditioning"],
                hidden_layers=self.hidden_layers,
                hidden_features=self.hidden_features,
                out_dim=self.out_features,
                outermost_linear=self.last_layer_linear,
                first_omega_0=self.first_omega_0,
                hidden_omega_0=self.hidden_omega_0,
                out_activation=output_activation,
            )
        elif self.conditioning == "FiLM":
            network = FiLMSiren(
                in_dim=input_dims["direction"],
                hidden_layers=self.hidden_layers,
                hidden_features=self.hidden_features,
                mapping_network_in_dim=input_dims["conditioning"],
                mapping_network_layers=self.mapping_layers,
                mapping_network_features=self.mapping_features,
                out_dim=self.out_features,
                outermost_linear=True,
                out_activation=output_activation,
            )
        elif self.conditioning == "Attention":
            # transformer where K, V is from conditioning input and Q is from pos encoded directional input
            network = Decoder(
                in_dim=input_dims["direction"],
                conditioning_input_dim=input_dims["conditioning"],
                hidden_features=self.cfg.hidden_features,
                num_heads=self.cfg.num_attention_heads,
                num_layers=self.cfg.num_attention_layers,
                out_activation=output_activation,
            )
        assert network is not None, "unknown conditioning type"
        return network

    def apply_positional_encoding(self, directional_input, conditioning_input):
        # conditioning on just invariant directional input
        if self.cfg.encoded_input == "Conditioning":
            conditioning_input = self.conditioning_encoding(
                conditioning_input
            )  # [num_rays, embedding_dim]
        elif self.cfg.encoded_input == "Directions":
            directional_input = self.direction_encoding(
                directional_input
            )  # [num_rays, embedding_dim]
        elif self.cfg.encoded_input == "Both":
            directional_input = self.direction_encoding(directional_input)
            conditioning_input = self.conditioning_encoding(conditioning_input)

        return directional_input, conditioning_input

    def get_outputs(
        self,
        rays_d: Float[Tensor, "batch num_rays 3"],  # type: ignore
        latent_codes: Float[Tensor, "batch_size latent_dim 3"],  # type: ignore
        rotation: Optional[Float[Tensor, "batch_size 3 3"]] = None,  # type: ignore
        scale: Optional[Float[Tensor, "batch_size"]] = None,  # type: ignore
    ) -> Dict[str, Tensor]:
        """Returns the outputs of the field.

        Args:
            ray_samples: [batch_size num_rays 3]
            latent_codes: [batch_size, latent_dim, 3]
            rotation: [batch_size, 3, 3]
            scale: [batch_size]
        """
        if rotation is not None:
            if len(rotation.shape) == 3:  # [batch_size, 3, 3]
                # Expand latent_codes to match [batch_size, latent_dim, 3]
                latent_codes = torch.einsum(
                    "bik,blk->bli",
                    rotation,
                    latent_codes,
                )
            else:
                raise NotImplementedError(
                    "Unsupported rotation shape. Expected [batch_size, 3, 3]."
                )

        B, num_rays, _ = rays_d.shape
        _, latent_dim, _ = latent_codes.shape

        if not self.old_implementation:
            directional_input, conditioning_input = self.invariant_function(
                latent_codes,
                rays_d,
                equivariance=self.equivariance,
                axis_of_invariance=self.axis_of_invariance,
            )  # [B, num_rays, 3]

            if self.cfg.positional_encoding == "NeRF":
                directional_input, conditioning_input = self.apply_positional_encoding(
                    directional_input, conditioning_input
                )

            if self.conditioning == "Concat":
                model_outputs = self.network(
                    torch.cat((directional_input, conditioning_input), dim=-1).reshape(
                        B * num_rays, -1
                    )
                ).view(B, num_rays, 3)  # returns -> [B num_rays, 3]
            elif self.conditioning == "FiLM":
                model_outputs = self.network(
                    directional_input.reshape(B * num_rays, -1),
                    conditioning_input.reshape(B * num_rays, -1),
                ).view(B, num_rays, 3)  # returns -> [B num_rays, 3]
            elif self.conditioning == "Attention":
                model_outputs = self.network(
                    directional_input.reshape(B * num_rays, -1),
                    conditioning_input.reshape(B * num_rays, -1),
                ).view(B, num_rays, 3)  # returns -> [B num_rays, 3]
        else:
            # in the old implementation directions were sampled with y-up not z-up so need to swap y and z in directions
            directions = torch.stack(
                (rays_d[..., 0], rays_d[..., 2], rays_d[..., 1]), -1
            )
            model_input = self.invariant_function(
                latent_codes,
                directions,
                equivariance=self.equivariance,
                axis_of_invariance=self.axis_of_invariance,
            )  # [B, num_rays, 3]

            model_outputs = self.network(model_input.view(B * num_rays, -1)).view(
                B, num_rays, 3
            )

        outputs = {}

        if scale is not None:
            scale = trunc_exp(scale)  # [num_rays] exp to ensure positive
            model_outputs = model_outputs * scale.view(-1, 1, 1)  # [num_rays, 3]

        outputs["rgb"] = model_outputs

        return outputs

    def forward(
        self,
        rays_d: Float[Tensor, "batch num_rays 3"],  # type: ignore
        latent_codes: Float[Tensor, "batch_size latent_dim 3"],  # type: ignore
        rotation: Optional[Float[Tensor, "batch_size 3 3"]] = None,  # type: ignore
        scale: Optional[Float[Tensor, "batch_size"]] = None,  # type: ignore
    ) -> Dict[str, Tensor]:
        """Evaluates spherical field for a given ray bundle and rotation.

        Args:
            ray_samples: [B num_rays 3]
            latent_codes: [B, num_rays, latent_dim, 3]
            rotation: [batch_size, 3, 3]
            scale: [batch_size]

        Returns:
            Dict[str, Tensor]: A dictionary containing the outputs of the field.
        """
        return self.get_outputs(
            rays_d=rays_d,
            latent_codes=latent_codes,
            rotation=rotation,
            scale=scale,
        )
