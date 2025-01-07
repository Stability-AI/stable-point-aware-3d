from typing import Optional

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        direction_input_dim: int,
        conditioning_input_dim: int,
        latent_dim: int,
        num_heads: int,
    ):
        """
        Multi-Head Attention module.

        Args:
            direction_input_dim (int): The input dimension of the directional input.
            conditioning_input_dim (int): The input dimension of the conditioning input.
            latent_dim (int): The latent dimension of the module.
            num_heads (int): The number of heads to use in the attention mechanism.
        """
        super().__init__()
        assert latent_dim % num_heads == 0, "latent_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.query = nn.Linear(direction_input_dim, latent_dim)
        self.key = nn.Linear(conditioning_input_dim, latent_dim)
        self.value = nn.Linear(conditioning_input_dim, latent_dim)
        self.fc_out = nn.Linear(latent_dim, latent_dim)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the Multi-Head Attention module.

        Args:
            query (torch.Tensor): The directional input tensor.
            key (torch.Tensor): The conditioning input tensor for the keys.
            value (torch.Tensor): The conditioning input tensor for the values.

        Returns:
            torch.Tensor: The output tensor of the Multi-Head Attention module.
        """
        batch_size = query.size(0)

        Q = (
            self.query(query)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.key(key)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.value(value)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attention = (
            torch.einsum("bnqk,bnkh->bnqh", [Q, K.transpose(-2, -1)]) * self.scale
        )
        attention = torch.softmax(attention, dim=-1)

        out = torch.einsum("bnqh,bnhv->bnqv", [attention, V])
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.head_dim)
        )

        out = self.fc_out(out).squeeze(1)
        return out


class AttentionLayer(nn.Module):
    def __init__(
        self,
        direction_input_dim: int,
        conditioning_input_dim: int,
        latent_dim: int,
        num_heads: int,
    ):
        """
        Attention Layer module.

        Args:
            direction_input_dim (int): The input dimension of the directional input.
            conditioning_input_dim (int): The input dimension of the conditioning input.
            latent_dim (int): The latent dimension of the module.
            num_heads (int): The number of heads to use in the attention mechanism.
        """
        super().__init__()
        self.mha = MultiHeadAttention(
            direction_input_dim, conditioning_input_dim, latent_dim, num_heads
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(
        self, directional_input: torch.Tensor, conditioning_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the Attention Layer module.

        Args:
            directional_input (torch.Tensor): The directional input tensor.
            conditioning_input (torch.Tensor): The conditioning input tensor.

        Returns:
            torch.Tensor: The output tensor of the Attention Layer module.
        """
        attn_output = self.mha(
            directional_input, conditioning_input, conditioning_input
        )
        out1 = self.norm1(attn_output + directional_input)
        fc_output = self.fc(out1)
        out2 = self.norm2(fc_output + out1)
        return out2


class Decoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        conditioning_input_dim: int,
        hidden_features: int,
        num_heads: int,
        num_layers: int,
        out_activation: Optional[nn.Module],
    ):
        """
        Decoder module.

        Args:
            in_dim (int): The input dimension of the module.
            conditioning_input_dim (int): The input dimension of the conditioning input.
            hidden_features (int): The number of hidden features in the module.
            num_heads (int): The number of heads to use in the attention mechanism.
            num_layers (int): The number of layers in the module.
            out_activation (nn.Module): The activation function to use on the output tensor.
        """
        super().__init__()
        self.residual_projection = nn.Linear(
            in_dim, hidden_features
        )  # projection for residual connection
        self.layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_features, conditioning_input_dim, hidden_features, num_heads
                )
                for i in range(num_layers)
            ]
        )
        self.fc = nn.Linear(hidden_features, 3)  # 3 for RGB
        self.out_activation = out_activation

    def forward(
        self, x: torch.Tensor, conditioning_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the Decoder module.

        Args:
            x (torch.Tensor): The input tensor.
            conditioning_input (torch.Tensor): The conditioning input tensor.

        Returns:
            torch.Tensor: The output tensor of the Decoder module.
        """
        x = self.residual_projection(x)
        for layer in self.layers:
            x = layer(x, conditioning_input)
        x = self.fc(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x
