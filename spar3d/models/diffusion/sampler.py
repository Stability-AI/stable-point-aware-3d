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

from typing import Dict, Iterator

import torch
import torch.nn as nn

from .gaussian_diffusion import GaussianDiffusion


class PointCloudSampler:
    """
    A wrapper around a model that produces conditional sample tensors.
    """

    def __init__(
        self,
        model: nn.Module,
        diffusion: GaussianDiffusion,
        num_points: int,
        point_dim: int = 3,
        guidance_scale: float = 3.0,
        clip_denoised: bool = True,
        sigma_min: float = 1e-3,
        sigma_max: float = 120,
        s_churn: float = 3,
    ):
        self.model = model
        self.num_points = num_points
        self.point_dim = point_dim
        self.guidance_scale = guidance_scale
        self.clip_denoised = clip_denoised
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.s_churn = s_churn

        self.diffusion = diffusion

    def sample_batch_progressive(
        self,
        batch_size: int,
        condition: torch.Tensor,
        noise=None,
        device=None,
        guidance_scale=None,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Generate samples progressively using classifier-free guidance.

        Args:
            batch_size: Number of samples to generate
            condition: Conditioning tensor
            noise: Optional initial noise tensor
            device: Device to run on
            guidance_scale: Optional override for guidance scale

        Returns:
            Iterator of dicts containing intermediate samples
        """
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        sample_shape = (batch_size, self.point_dim, self.num_points)

        # Double the batch for classifier-free guidance
        if guidance_scale != 1 and guidance_scale != 0:
            condition = torch.cat([condition, torch.zeros_like(condition)], dim=0)
            if noise is not None:
                noise = torch.cat([noise, noise], dim=0)
        model_kwargs = {"condition": condition}

        internal_batch_size = batch_size
        if guidance_scale != 1 and guidance_scale != 0:
            model = self._uncond_guide_model(self.model, guidance_scale)
            internal_batch_size *= 2
        else:
            model = self.model

        samples_it = self.diffusion.ddim_sample_loop_progressive(
            model,
            shape=(internal_batch_size, *sample_shape[1:]),
            model_kwargs=model_kwargs,
            device=device,
            clip_denoised=self.clip_denoised,
            noise=noise,
        )

        for x in samples_it:
            samples = {
                "xstart": x["pred_xstart"][:batch_size],
                "xprev": x["sample"][:batch_size] if "sample" in x else x["x"],
            }
            yield samples

    def _uncond_guide_model(self, model: nn.Module, scale: float) -> nn.Module:
        """
        Wraps the model for classifier-free guidance.
        """

        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = model(combined, ts, **kwargs)

            eps, rest = model_out[:, : self.point_dim], model_out[:, self.point_dim :]
            cond_eps, uncond_eps = torch.chunk(eps, 2, dim=0)
            half_eps = uncond_eps + scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        return model_fn
