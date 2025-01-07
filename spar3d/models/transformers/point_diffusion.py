# --------------------------------------------------------
# Adapted from: https://github.com/openai/point-e
# Licensed under the MIT License
# Copyright (c) 2022 OpenAI

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# --------------------------------------------------------

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn

from spar3d.models.utils import BaseModule


def init_linear(layer, stddev):
    nn.init.normal_(layer.weight, std=stddev)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        init_scale: float,
    ):
        super().__init__()
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3)
        self.c_proj = nn.Linear(width, width)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        bs, n_ctx, width = x.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(attn_ch)
        x = x.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(x, attn_ch, dim=-1)

        x = (
            torch.nn.functional.scaled_dot_product_attention(
                q.permute(0, 2, 1, 3),
                k.permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3),
                scale=scale,
            )
            .permute(0, 2, 1, 3)
            .reshape(bs, n_ctx, -1)
        )

        x = self.c_proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, *, width: int, init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class ResidualAttentionBlock(nn.Module):
    def __init__(self, *, width: int, heads: int, init_scale: float = 1.0):
        super().__init__()

        self.attn = MultiheadAttention(
            width=width,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = MLP(width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x


class PointDiffusionTransformer(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int = 3,
        output_channels: int = 3,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        init_scale: float = 0.25,
        time_token_cond: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.time_token_cond = time_token_cond
        self.time_embed = MLP(
            width=width,
            init_scale=init_scale * math.sqrt(1.0 / width),
        )
        self.ln_pre = nn.LayerNorm(width)
        self.backbone = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_post = nn.LayerNorm(width)
        self.input_proj = nn.Linear(input_channels, width)
        self.output_proj = nn.Linear(width, output_channels)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :return: an [N x C' x T] tensor.
        """
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        return self._forward_with_cond(x, [(t_embed, self.time_token_cond)])

    def _forward_with_cond(
        self, x: torch.Tensor, cond_as_token: List[Tuple[torch.Tensor, bool]]
    ) -> torch.Tensor:
        h = self.input_proj(x.permute(0, 2, 1))  # NCL -> NLC
        for emb, as_token in cond_as_token:
            if not as_token:
                h = h + emb[:, None]
        extra_tokens = [
            (emb[:, None] if len(emb.shape) == 2 else emb)
            for emb, as_token in cond_as_token
            if as_token
        ]
        if len(extra_tokens):
            h = torch.cat(extra_tokens + [h], dim=1)

        h = self.ln_pre(h)
        h = self.backbone(h)
        h = self.ln_post(h)
        if len(extra_tokens):
            h = h[:, sum(h.shape[1] for h in extra_tokens) :]
        h = self.output_proj(h)
        return h.permute(0, 2, 1)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class PointEDenoiser(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        num_attention_heads: int = 8
        in_channels: Optional[int] = None
        out_channels: Optional[int] = None
        num_layers: int = 12
        width: int = 512
        cond_dim: Optional[int] = None

    cfg: Config

    def configure(self) -> None:
        self.denoiser = PointDiffusionTransformer(
            input_channels=self.cfg.in_channels,
            output_channels=self.cfg.out_channels,
            width=self.cfg.width,
            layers=self.cfg.num_layers,
            heads=self.cfg.num_attention_heads,
            init_scale=0.25,
            time_token_cond=True,
        )

        self.cond_embed = nn.Sequential(
            nn.LayerNorm(self.cfg.cond_dim),
            nn.Linear(self.cfg.cond_dim, self.cfg.width),
        )

    def forward(
        self,
        x,
        t,
        condition=None,
    ):
        # renormalize with the per-sample standard deviation
        x_std = torch.std(x.reshape(x.shape[0], -1), dim=1, keepdim=True)
        x = x / x_std.reshape(-1, *([1] * (len(x.shape) - 1)))

        t_embed = self.denoiser.time_embed(
            timestep_embedding(t, self.denoiser.backbone.width)
        )
        condition = self.cond_embed(condition)

        cond = [(t_embed, True), (condition, True)]
        x_denoised = self.denoiser._forward_with_cond(x, cond)
        return x_denoised
