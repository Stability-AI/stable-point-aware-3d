from dataclasses import dataclass
from typing import Optional

import torch
from jaxtyping import Float
from torch import Tensor

from spar3d.models.transformers.transformer_1d import Transformer1D
from spar3d.models.utils import BaseModule


class TransformerPointTokenizer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        num_attention_heads: int = 16
        attention_head_dim: int = 64
        in_channels: Optional[int] = 6
        out_channels: Optional[int] = 1024
        num_layers: int = 16
        norm_num_groups: int = 32
        attention_bias: bool = False
        activation_fn: str = "geglu"
        norm_elementwise_affine: bool = True

    cfg: Config

    def configure(self) -> None:
        transformer_cfg = dict(self.cfg.copy())
        # remove the non-transformer configs
        transformer_cfg["in_channels"] = (
            self.cfg.num_attention_heads * self.cfg.attention_head_dim
        )
        self.model = Transformer1D(transformer_cfg)
        self.linear_in = torch.nn.Linear(
            self.cfg.in_channels, transformer_cfg["in_channels"]
        )
        self.linear_out = torch.nn.Linear(
            transformer_cfg["in_channels"], self.cfg.out_channels
        )

    def forward(
        self, points: Float[Tensor, "B N Ci"], **kwargs
    ) -> Float[Tensor, "B N Cp"]:
        assert points.ndim == 3
        inputs = self.linear_in(points).permute(0, 2, 1)  # B N Ci -> B Ci N
        out = self.model(inputs).permute(0, 2, 1)  # B Ci N -> B N Ci
        out = self.linear_out(out)  # B N Ci -> B N Co
        return out

    def detokenize(self, *args, **kwargs):
        raise NotImplementedError
