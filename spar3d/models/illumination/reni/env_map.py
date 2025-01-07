from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from jaxtyping import Float
from torch import Tensor

from spar3d.models.utils import BaseModule

from .field import RENIField


def _direction_from_coordinate(
    coordinate: Float[Tensor, "*B 2"],
) -> Float[Tensor, "*B 3"]:
    # OpenGL Convention
    # +X Right
    # +Y Up
    # +Z Backward

    u, v = coordinate.unbind(-1)
    theta = (2 * torch.pi * u) - torch.pi
    phi = torch.pi * v

    dir = torch.stack(
        [
            theta.sin() * phi.sin(),
            phi.cos(),
            -1 * theta.cos() * phi.sin(),
        ],
        -1,
    )
    return dir


def _get_sample_coordinates(
    resolution: List[int], device: Optional[torch.device] = None
) -> Float[Tensor, "H W 2"]:
    return torch.stack(
        torch.meshgrid(
            (torch.arange(resolution[1], device=device) + 0.5) / resolution[1],
            (torch.arange(resolution[0], device=device) + 0.5) / resolution[0],
            indexing="xy",
        ),
        -1,
    )


class RENIEnvMap(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        reni_config: dict = field(default_factory=dict)
        resolution: int = 128

    cfg: Config

    def configure(self):
        self.field = RENIField(self.cfg.reni_config)
        resolution = (self.cfg.resolution, self.cfg.resolution * 2)
        sample_directions = _direction_from_coordinate(
            _get_sample_coordinates(resolution)
        )
        self.img_shape = sample_directions.shape[:-1]

        sample_directions_flat = sample_directions.view(-1, 3)
        # Lastly these have y up but reni expects z up. Rotate 90 degrees on x axis
        sample_directions_flat = torch.stack(
            [
                sample_directions_flat[:, 0],
                -sample_directions_flat[:, 2],
                sample_directions_flat[:, 1],
            ],
            -1,
        )
        self.sample_directions = torch.nn.Parameter(
            sample_directions_flat, requires_grad=False
        )

    def forward(
        self,
        latent_codes: Float[Tensor, "B latent_dim 3"],
        rotation: Optional[Float[Tensor, "B 3 3"]] = None,
        scale: Optional[Float[Tensor, "B"]] = None,
    ) -> Dict[str, Tensor]:
        return {
            k: v.view(latent_codes.shape[0], *self.img_shape, -1)
            for k, v in self.field(
                self.sample_directions.unsqueeze(0).repeat(latent_codes.shape[0], 1, 1),
                latent_codes,
                rotation=rotation,
                scale=scale,
            ).items()
        }
