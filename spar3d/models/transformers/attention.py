from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Modulation(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        condition_dim: int,
        zero_init: bool = False,
        single_layer: bool = False,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        if single_layer:
            self.linear1 = nn.Identity()
        else:
            self.linear1 = nn.Linear(condition_dim, condition_dim)

        self.linear2 = nn.Linear(condition_dim, embedding_dim * 2)

        # Only zero init the last linear layer
        if zero_init:
            nn.init.zeros_(self.linear2.weight)
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        emb = self.linear2(self.silu(self.linear1(condition)))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        linear_cls = nn.Linear

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh")
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(linear_cls(inner_dim, dim_out))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        out_bias: bool = True,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.num_heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout

        # Linear projections
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=bias)

        # Output projection
        self.to_out = nn.ModuleList(
            [
                nn.Linear(self.inner_dim, query_dim, bias=out_bias),
                nn.Dropout(dropout),
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        # Project queries, keys, and values
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # Reshape for multi-head attention
        query = query.reshape(
            batch_size, sequence_length, self.num_heads, -1
        ).transpose(1, 2)
        key = key.reshape(batch_size, sequence_length, self.num_heads, -1).transpose(
            1, 2
        )
        value = value.reshape(
            batch_size, sequence_length, self.num_heads, -1
        ).transpose(1, 2)

        # Compute scaled dot product attention
        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            scale=self.scale,
        )

        # Reshape and project output
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, sequence_length, self.inner_dim
        )

        # Apply output projection and dropout
        for module in self.to_out:
            hidden_states = module(hidden_states)

        return hidden_states


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        # Self-Attn
        self.norm1 = nn.LayerNorm(
            dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
        )
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
        )

        # Feed-forward
        self.norm3 = nn.LayerNorm(
            dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
        )
        self.ff = FeedForward(
            dim,
            activation_fn=activation_fn,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # Self-Attention
        norm_hidden_states = self.norm1(hidden_states)

        hidden_states = (
            self.attn1(
                norm_hidden_states,
                attention_mask=attention_mask,
            )
            + hidden_states
        )

        # Feed-forward
        ff_output = self.ff(self.norm3(hidden_states))

        hidden_states = ff_output + hidden_states

        return hidden_states


class GELU(nn.Module):
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
    """

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
        self.approximate = approximate

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type != "mps":
            return F.gelu(gate, approximate=self.approximate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(
            dtype=gate.dtype
        )

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        linear_cls = nn.Linear

        self.proj = linear_cls(dim_in, dim_out * 2)

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type != "mps":
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states, scale: float = 1.0):
        args = ()
        hidden_states, gate = self.proj(hidden_states, *args).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class ApproximateGELU(nn.Module):
    r"""
    The approximate form of Gaussian Error Linear Unit (GELU). For more details, see section 2:
    https://arxiv.org/abs/1606.08415.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)
