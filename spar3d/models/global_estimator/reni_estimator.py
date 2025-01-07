from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from spar3d.models.illumination.reni.env_map import RENIEnvMap
from spar3d.models.utils import BaseModule


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    assert d6.shape[-1] == 6, "Input tensor must have shape (..., 6)"

    def proj_u2a(u, a):
        r"""
        u: batch x 3
        a: batch x 3
        """
        inner_prod = torch.sum(u * a, dim=-1, keepdim=True)
        norm2 = torch.sum(u**2, dim=-1, keepdim=True)
        norm2 = torch.clamp(norm2, min=1e-8)
        factor = inner_prod / (norm2 + 1e-10)
        return factor * u

    x_raw, y_raw = d6[..., :3], d6[..., 3:]

    x = F.normalize(x_raw, dim=-1)
    y = F.normalize(y_raw - proj_u2a(x, y_raw), dim=-1)
    z = torch.cross(x, y, dim=-1)

    return torch.stack((x, y, z), dim=-1)


class ReniLatentCodeEstimator(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        triplane_features: int = 40

        n_layers: int = 5
        hidden_features: int = 512
        activation: str = "relu"

        pool: str = "mean"

        reni_env_config: dict = field(default_factory=dict)

    cfg: Config

    def configure(self):
        layers = []
        cur_features = self.cfg.triplane_features * 3
        for _ in range(self.cfg.n_layers):
            layers.append(
                nn.Conv2d(
                    cur_features,
                    self.cfg.hidden_features,
                    kernel_size=3,
                    padding=0,
                    stride=2,
                )
            )
            layers.append(self.make_activation(self.cfg.activation))

            cur_features = self.cfg.hidden_features

        self.layers = nn.Sequential(*layers)

        self.reni_env_map = RENIEnvMap(self.cfg.reni_env_config)
        self.latent_dim = self.reni_env_map.field.latent_dim

        self.fc_latents = nn.Linear(self.cfg.hidden_features, self.latent_dim * 3)
        nn.init.normal_(self.fc_latents.weight, mean=0.0, std=0.3)

        self.fc_rotations = nn.Linear(self.cfg.hidden_features, 6)
        nn.init.constant_(self.fc_rotations.bias, 0.0)
        nn.init.normal_(
            self.fc_rotations.weight, mean=0.0, std=0.01
        )  # Small variance here

        self.fc_scale = nn.Linear(self.cfg.hidden_features, 1)
        nn.init.constant_(self.fc_scale.bias, 0.0)
        nn.init.normal_(self.fc_scale.weight, mean=0.0, std=0.01)  # Small variance here

    def make_activation(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        else:
            raise NotImplementedError

    def forward(
        self,
        triplane: Float[Tensor, "B 3 F Ht Wt"],
        rotation: Optional[Float[Tensor, "B 3 3"]] = None,
    ) -> dict[str, Any]:
        x = self.layers(
            triplane.reshape(
                triplane.shape[0], -1, triplane.shape[-2], triplane.shape[-1]
            )
        )
        x = x.mean(dim=[-2, -1])

        latents = self.fc_latents(x).reshape(-1, self.latent_dim, 3)
        rotations = rotation_6d_to_matrix(self.fc_rotations(x))
        scale = self.fc_scale(x)

        if rotation is not None:
            rotations = rotations @ rotation.to(dtype=rotations.dtype)

        env_map = self.reni_env_map(latents, rotations, scale)

        return {"illumination": env_map["rgb"]}
