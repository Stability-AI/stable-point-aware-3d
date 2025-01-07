import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from einops import rearrange
from huggingface_hub import hf_hub_download
from jaxtyping import Float
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file, load_model
from torch import Tensor

from spar3d.models.diffusion.gaussian_diffusion import (
    SpacedDiffusion,
    get_named_beta_schedule,
    space_timesteps,
)
from spar3d.models.diffusion.sampler import PointCloudSampler
from spar3d.models.isosurface import MarchingTetrahedraHelper
from spar3d.models.mesh import Mesh
from spar3d.models.utils import (
    BaseModule,
    ImageProcessor,
    convert_data,
    dilate_fill,
    find_class,
    float32_to_uint8_np,
    normalize,
    scale_tensor,
)
from spar3d.utils import (
    create_intrinsic_from_fov_rad,
    default_cond_c2w,
    get_device,
    normalize_pc_bbox,
)

try:
    from texture_baker import TextureBaker
except ImportError:
    import logging

    logging.warning(
        "Could not import texture_baker. Please install it via `pip install texture-baker/`"
    )
    # Exit early to avoid further errors
    raise ImportError("texture_baker not found")


class SPAR3D(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        cond_image_size: int
        isosurface_resolution: int
        isosurface_threshold: float = 10.0
        radius: float = 1.0
        background_color: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
        default_fovy_rad: float = 0.591627
        default_distance: float = 2.2

        camera_embedder_cls: str = ""
        camera_embedder: dict = field(default_factory=dict)

        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)

        point_embedder_cls: str = ""
        point_embedder: dict = field(default_factory=dict)

        tokenizer_cls: str = ""
        tokenizer: dict = field(default_factory=dict)

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        post_processor_cls: str = ""
        post_processor: dict = field(default_factory=dict)

        decoder_cls: str = ""
        decoder: dict = field(default_factory=dict)

        image_estimator_cls: str = ""
        image_estimator: dict = field(default_factory=dict)

        global_estimator_cls: str = ""
        global_estimator: dict = field(default_factory=dict)

        # Point diffusion modules
        pdiff_camera_embedder_cls: str = ""
        pdiff_camera_embedder: dict = field(default_factory=dict)

        pdiff_image_tokenizer_cls: str = ""
        pdiff_image_tokenizer: dict = field(default_factory=dict)

        pdiff_backbone_cls: str = ""
        pdiff_backbone: dict = field(default_factory=dict)

        scale_factor_xyz: float = 1.0
        scale_factor_rgb: float = 1.0
        bias_xyz: float = 0.0
        bias_rgb: float = 0.0
        train_time_steps: int = 1024
        inference_time_steps: int = 64

        mean_type: str = "epsilon"
        var_type: str = "fixed_small"
        diffu_sched: str = "cosine"
        diffu_sched_exp: float = 12.0
        guidance_scale: float = 3.0
        sigma_max: float = 120.0
        s_churn: float = 3.0

        low_vram_mode: bool = False

    cfg: Config

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config_name: str,
        weight_name: str,
        low_vram_mode: bool = False,
    ):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if os.path.isdir(os.path.join(base_dir, pretrained_model_name_or_path)):
            config_path = os.path.join(
                base_dir, pretrained_model_name_or_path, config_name
            )
            weight_path = os.path.join(
                base_dir, pretrained_model_name_or_path, weight_name
            )
        else:
            config_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=config_name
            )
            weight_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=weight_name
            )

        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        # Add in low_vram_mode to the config
        if os.environ.get("SPAR3D_LOW_VRAM", "0") == "1" and torch.cuda.is_available():
            cfg.low_vram_mode = True
        else:
            cfg.low_vram_mode = low_vram_mode if torch.cuda.is_available() else False
        model = cls(cfg)

        if not model.cfg.low_vram_mode:
            load_model(model, weight_path, strict=False)
        else:
            model._state_dict = load_file(weight_path, device="cpu")

        return model

    @property
    def device(self):
        return next(self.parameters()).device

    def configure(self):
        # Initialize all modules as None
        self.image_tokenizer = None
        self.point_embedder = None
        self.tokenizer = None
        self.camera_embedder = None
        self.backbone = None
        self.post_processor = None
        self.decoder = None
        self.image_estimator = None
        self.global_estimator = None
        self.pdiff_image_tokenizer = None
        self.pdiff_camera_embedder = None
        self.pdiff_backbone = None
        self.diffusion_spaced = None
        self.sampler = None

        # Dummy parameter to safe the device placement for dynamic loading
        self.dummy_param = torch.nn.Parameter(torch.tensor(0.0))

        channel_scales = [self.cfg.scale_factor_xyz] * 3
        channel_scales += [self.cfg.scale_factor_rgb] * 3
        channel_biases = [self.cfg.bias_xyz] * 3
        channel_biases += [self.cfg.bias_rgb] * 3
        channel_scales = np.array(channel_scales)
        channel_biases = np.array(channel_biases)

        betas = get_named_beta_schedule(
            self.cfg.diffu_sched, self.cfg.train_time_steps, self.cfg.diffu_sched_exp
        )

        self.diffusion_kwargs = dict(
            betas=betas,
            model_mean_type=self.cfg.mean_type,
            model_var_type=self.cfg.var_type,
            channel_scales=channel_scales,
            channel_biases=channel_biases,
        )

        self.is_low_vram = self.cfg.low_vram_mode and get_device() == "cuda"

        # Create CPU shadow copy if in low VRAM mode
        if not self.is_low_vram:
            self._load_all_modules()
        else:
            print("Loading in low VRAM mode")

        self.bbox: Float[Tensor, "2 3"]
        self.register_buffer(
            "bbox",
            torch.as_tensor(
                [
                    [-self.cfg.radius, -self.cfg.radius, -self.cfg.radius],
                    [self.cfg.radius, self.cfg.radius, self.cfg.radius],
                ],
                dtype=torch.float32,
            ),
        )
        self.isosurface_helper = MarchingTetrahedraHelper(
            self.cfg.isosurface_resolution,
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "load",
                "tets",
                f"{self.cfg.isosurface_resolution}_tets.npz",
            ),
        )

        self.baker = TextureBaker()
        self.image_processor = ImageProcessor()

    def _load_all_modules(self):
        """Load all modules into memory"""
        # Load modules to specified device
        self.image_tokenizer = find_class(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        ).to(self.device)
        self.point_embedder = find_class(self.cfg.point_embedder_cls)(
            self.cfg.point_embedder
        ).to(self.device)
        self.tokenizer = find_class(self.cfg.tokenizer_cls)(self.cfg.tokenizer).to(
            self.device
        )
        self.camera_embedder = find_class(self.cfg.camera_embedder_cls)(
            self.cfg.camera_embedder
        ).to(self.device)
        self.backbone = find_class(self.cfg.backbone_cls)(self.cfg.backbone).to(
            self.device
        )
        self.post_processor = find_class(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        ).to(self.device)
        self.decoder = find_class(self.cfg.decoder_cls)(self.cfg.decoder).to(
            self.device
        )
        self.image_estimator = find_class(self.cfg.image_estimator_cls)(
            self.cfg.image_estimator
        ).to(self.device)
        self.global_estimator = find_class(self.cfg.global_estimator_cls)(
            self.cfg.global_estimator
        ).to(self.device)
        self.pdiff_image_tokenizer = find_class(self.cfg.pdiff_image_tokenizer_cls)(
            self.cfg.pdiff_image_tokenizer
        ).to(self.device)
        self.pdiff_camera_embedder = find_class(self.cfg.pdiff_camera_embedder_cls)(
            self.cfg.pdiff_camera_embedder
        ).to(self.device)
        self.pdiff_backbone = find_class(self.cfg.pdiff_backbone_cls)(
            self.cfg.pdiff_backbone
        ).to(self.device)

        self.diffusion_spaced = SpacedDiffusion(
            use_timesteps=space_timesteps(
                self.cfg.train_time_steps,
                "ddim" + str(self.cfg.inference_time_steps),
            ),
            **self.diffusion_kwargs,
        )
        self.sampler = PointCloudSampler(
            model=self.pdiff_backbone,
            diffusion=self.diffusion_spaced,
            num_points=512,
            point_dim=6,
            guidance_scale=self.cfg.guidance_scale,
            clip_denoised=True,
            sigma_min=1e-3,
            sigma_max=self.cfg.sigma_max,
            s_churn=self.cfg.s_churn,
        )

    def _load_main_modules(self):
        """Load the main processing modules"""
        if all(
            [
                self.image_tokenizer,
                self.point_embedder,
                self.tokenizer,
                self.camera_embedder,
                self.backbone,
                self.post_processor,
                self.decoder,
            ]
        ):
            return  # Main modules already loaded

        device = next(self.parameters()).device  # Get the current device

        self.image_tokenizer = find_class(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        ).to(device)
        self.point_embedder = find_class(self.cfg.point_embedder_cls)(
            self.cfg.point_embedder
        ).to(device)
        self.tokenizer = find_class(self.cfg.tokenizer_cls)(self.cfg.tokenizer).to(
            device
        )
        self.camera_embedder = find_class(self.cfg.camera_embedder_cls)(
            self.cfg.camera_embedder
        ).to(device)
        self.backbone = find_class(self.cfg.backbone_cls)(self.cfg.backbone).to(device)
        self.post_processor = find_class(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        ).to(device)
        self.decoder = find_class(self.cfg.decoder_cls)(self.cfg.decoder).to(device)

        # Restore weights if we have a checkpoint path
        if hasattr(self, "_state_dict"):
            self.load_state_dict(self._state_dict, strict=False)

    def _load_estimator_modules(self):
        """Load the estimator modules"""
        if all([self.image_estimator, self.global_estimator]):
            return  # Estimator modules already loaded

        device = next(self.parameters()).device  # Get the current device

        self.image_estimator = find_class(self.cfg.image_estimator_cls)(
            self.cfg.image_estimator
        ).to(device)
        self.global_estimator = find_class(self.cfg.global_estimator_cls)(
            self.cfg.global_estimator
        ).to(device)

        # Restore weights if we have a checkpoint path
        if hasattr(self, "_state_dict"):
            self.load_state_dict(self._state_dict, strict=False)

    def _load_pdiff_modules(self):
        """Load only the point diffusion modules"""
        if all(
            [
                self.pdiff_image_tokenizer,
                self.pdiff_camera_embedder,
                self.pdiff_backbone,
            ]
        ):
            return  # PDiff modules already loaded

        device = next(self.parameters()).device  # Get the current device

        self.pdiff_image_tokenizer = find_class(self.cfg.pdiff_image_tokenizer_cls)(
            self.cfg.pdiff_image_tokenizer
        ).to(device)
        self.pdiff_camera_embedder = find_class(self.cfg.pdiff_camera_embedder_cls)(
            self.cfg.pdiff_camera_embedder
        ).to(device)
        self.pdiff_backbone = find_class(self.cfg.pdiff_backbone_cls)(
            self.cfg.pdiff_backbone
        ).to(device)

        self.diffusion_spaced = SpacedDiffusion(
            use_timesteps=space_timesteps(
                self.cfg.train_time_steps,
                "ddim" + str(self.cfg.inference_time_steps),
            ),
            **self.diffusion_kwargs,
        )
        self.sampler = PointCloudSampler(
            model=self.pdiff_backbone,
            diffusion=self.diffusion_spaced,
            num_points=512,
            point_dim=6,
            guidance_scale=self.cfg.guidance_scale,
            clip_denoised=True,
            sigma_min=1e-3,
            sigma_max=self.cfg.sigma_max,
            s_churn=self.cfg.s_churn,
        )

        # Restore weights if we have a checkpoint path
        if hasattr(self, "_state_dict"):
            self.load_state_dict(self._state_dict, strict=False)

    def _unload_pdiff_modules(self):
        """Unload point diffusion modules to free memory"""
        self.pdiff_image_tokenizer = None
        self.pdiff_camera_embedder = None
        self.pdiff_backbone = None
        self.diffusion_spaced = None
        self.sampler = None
        torch.cuda.empty_cache()

    def _unload_main_modules(self):
        """Unload main processing modules to free memory"""
        self.image_tokenizer = None
        self.point_embedder = None
        self.tokenizer = None
        self.camera_embedder = None
        self.backbone = None
        self.post_processor = None
        torch.cuda.empty_cache()

    def _unload_estimator_modules(self):
        """Unload estimator modules to free memory"""
        self.image_estimator = None
        self.global_estimator = None
        torch.cuda.empty_cache()

    def triplane_to_meshes(
        self, triplanes: Float[Tensor, "B 3 Cp Hp Wp"]
    ) -> list[Mesh]:
        meshes = []
        for i in range(triplanes.shape[0]):
            triplane = triplanes[i]
            grid_vertices = scale_tensor(
                self.isosurface_helper.grid_vertices.to(triplanes.device),
                self.isosurface_helper.points_range,
                self.bbox,
            )

            values = self.query_triplane(grid_vertices, triplane)
            decoded = self.decoder(values, include=["vertex_offset", "density"])
            sdf = decoded["density"] - self.cfg.isosurface_threshold

            deform = decoded["vertex_offset"].squeeze(0)

            mesh: Mesh = self.isosurface_helper(
                sdf.view(-1, 1), deform.view(-1, 3) if deform is not None else None
            )
            mesh.v_pos = scale_tensor(
                mesh.v_pos, self.isosurface_helper.points_range, self.bbox
            )

            meshes.append(mesh)

        return meshes

    def query_triplane(
        self,
        positions: Float[Tensor, "*B N 3"],
        triplanes: Float[Tensor, "*B 3 Cp Hp Wp"],
    ) -> Float[Tensor, "*B N F"]:
        batched = positions.ndim == 3
        if not batched:
            # no batch dimension
            triplanes = triplanes[None, ...]
            positions = positions[None, ...]
        assert triplanes.ndim == 5 and positions.ndim == 3

        positions = scale_tensor(
            positions, (-self.cfg.radius, self.cfg.radius), (-1, 1)
        )

        indices2D: Float[Tensor, "B 3 N 2"] = torch.stack(
            (positions[..., [0, 1]], positions[..., [0, 2]], positions[..., [1, 2]]),
            dim=-3,
        ).to(triplanes.dtype)
        out: Float[Tensor, "B3 Cp 1 N"] = F.grid_sample(
            rearrange(triplanes, "B Np Cp Hp Wp -> (B Np) Cp Hp Wp", Np=3).float(),
            rearrange(indices2D, "B Np N Nd -> (B Np) () N Nd", Np=3).float(),
            align_corners=True,
            mode="bilinear",
        )
        out = rearrange(out, "(B Np) Cp () N -> B N (Np Cp)", Np=3)

        return out

    def get_scene_codes(self, batch) -> Float[Tensor, "B 3 C H W"]:
        if self.is_low_vram:
            self._unload_pdiff_modules()
            self._unload_estimator_modules()
            self._load_main_modules()

        # if batch[rgb_cond] is only one view, add a view dimension
        if len(batch["rgb_cond"].shape) == 4:
            batch["rgb_cond"] = batch["rgb_cond"].unsqueeze(1)
            batch["mask_cond"] = batch["mask_cond"].unsqueeze(1)
            batch["c2w_cond"] = batch["c2w_cond"].unsqueeze(1)
            batch["intrinsic_cond"] = batch["intrinsic_cond"].unsqueeze(1)
            batch["intrinsic_normed_cond"] = batch["intrinsic_normed_cond"].unsqueeze(1)

        batch_size, n_input_views = batch["rgb_cond"].shape[:2]

        camera_embeds: Optional[Float[Tensor, "B Nv Cc"]]
        camera_embeds = self.camera_embedder(**batch)

        pc_embeds = self.point_embedder(batch["pc_cond"])

        input_image_tokens: Float[Tensor, "B Nv Cit Nit"] = self.image_tokenizer(
            rearrange(batch["rgb_cond"], "B Nv H W C -> B Nv C H W"),
            modulation_cond=camera_embeds,
        )

        input_image_tokens = rearrange(
            input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=n_input_views
        )

        tokens: Float[Tensor, "B Ct Nt"] = self.tokenizer(batch_size)

        cross_tokens = input_image_tokens
        cross_tokens = torch.cat([cross_tokens, pc_embeds], dim=1)

        tokens = self.backbone(
            tokens,
            encoder_hidden_states=cross_tokens,
            modulation_cond=None,
        )

        direct_codes = self.tokenizer.detokenize(tokens)
        scene_codes = self.post_processor(direct_codes)

        return scene_codes, direct_codes

    def forward_pdiff_cond(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.is_low_vram:
            self._unload_main_modules()
            self._unload_estimator_modules()
            self._load_pdiff_modules()

        if len(batch["rgb_cond"].shape) == 4:
            batch["rgb_cond"] = batch["rgb_cond"].unsqueeze(1)
            batch["mask_cond"] = batch["mask_cond"].unsqueeze(1)
            batch["c2w_cond"] = batch["c2w_cond"].unsqueeze(1)
            batch["intrinsic_cond"] = batch["intrinsic_cond"].unsqueeze(1)
            batch["intrinsic_normed_cond"] = batch["intrinsic_normed_cond"].unsqueeze(1)

        _batch_size, n_input_views = batch["rgb_cond"].shape[:2]

        # Camera modulation
        camera_embeds: Float[Tensor, "B Nv Cc"] = self.pdiff_camera_embedder(**batch)

        input_image_tokens: Float[Tensor, "B Nv Cit Nit"] = self.pdiff_image_tokenizer(
            rearrange(batch["rgb_cond"], "B Nv H W C -> B Nv C H W"),
            modulation_cond=camera_embeds,
        )

        input_image_tokens = rearrange(
            input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=n_input_views
        )

        return input_image_tokens

    def run_image(
        self,
        image: Union[Image.Image, List[Image.Image]],
        bake_resolution: int,
        pointcloud: Optional[Union[List[np.ndarray], np.ndarray, Tensor]] = None,
        remesh: Literal["none", "triangle", "quad"] = "none",
        vertex_count: int = -1,
        estimate_illumination: bool = False,
        return_points: bool = False,
    ) -> Tuple[Union[trimesh.Trimesh, List[trimesh.Trimesh]], dict[str, Any]]:
        if isinstance(image, list):
            rgb_cond = []
            mask_cond = []
            for img in image:
                mask, rgb = self.prepare_image(img)
                mask_cond.append(mask)
                rgb_cond.append(rgb)
            rgb_cond = torch.stack(rgb_cond, 0)
            mask_cond = torch.stack(mask_cond, 0)
            batch_size = rgb_cond.shape[0]
        else:
            mask_cond, rgb_cond = self.prepare_image(image)
            batch_size = 1

        c2w_cond = default_cond_c2w(self.cfg.default_distance).to(self.device)
        intrinsic, intrinsic_normed_cond = create_intrinsic_from_fov_rad(
            self.cfg.default_fovy_rad,
            self.cfg.cond_image_size,
            self.cfg.cond_image_size,
        )

        batch = {
            "rgb_cond": rgb_cond,
            "mask_cond": mask_cond,
            "c2w_cond": c2w_cond.view(1, 1, 4, 4).repeat(batch_size, 1, 1, 1),
            "intrinsic_cond": intrinsic.to(self.device)
            .view(1, 1, 3, 3)
            .repeat(batch_size, 1, 1, 1),
            "intrinsic_normed_cond": intrinsic_normed_cond.to(self.device)
            .view(1, 1, 3, 3)
            .repeat(batch_size, 1, 1, 1),
        }

        meshes, global_dict = self.generate_mesh(
            batch,
            bake_resolution,
            pointcloud,
            remesh,
            vertex_count,
            estimate_illumination,
        )

        if return_points:
            point_clouds = []
            for i in range(batch_size):
                xyz = batch["pc_cond"][i, :, :3].cpu().numpy()
                color_rgb = (
                    (batch["pc_cond"][i, :, 3:6] * 255).cpu().numpy().astype(np.uint8)
                )
                pc_trimesh = trimesh.PointCloud(vertices=xyz, colors=color_rgb)
                point_clouds.append(pc_trimesh)
            global_dict["point_clouds"] = point_clouds

        if batch_size == 1:
            return meshes[0], global_dict
        else:
            return meshes, global_dict

    def prepare_image(self, image):
        if image.mode != "RGBA":
            raise ValueError("Image must be in RGBA mode")
        img_cond = (
            torch.from_numpy(
                np.asarray(
                    image.resize((self.cfg.cond_image_size, self.cfg.cond_image_size))
                ).astype(np.float32)
                / 255.0
            )
            .float()
            .clip(0, 1)
            .to(self.device)
        )
        mask_cond = img_cond[:, :, -1:]
        rgb_cond = torch.lerp(
            torch.tensor(self.cfg.background_color, device=self.device)[None, None, :],
            img_cond[:, :, :3],
            mask_cond,
        )

        return mask_cond, rgb_cond

    def generate_mesh(
        self,
        batch,
        bake_resolution: int,
        pointcloud: Optional[Union[List[float], np.ndarray, Tensor]] = None,
        remesh: Literal["none", "triangle", "quad"] = "none",
        vertex_count: int = -1,
        estimate_illumination: bool = False,
    ) -> Tuple[List[trimesh.Trimesh], dict[str, Any]]:
        batch["rgb_cond"] = self.image_processor(
            batch["rgb_cond"], self.cfg.cond_image_size
        )
        batch["mask_cond"] = self.image_processor(
            batch["mask_cond"], self.cfg.cond_image_size
        )

        batch_size = batch["rgb_cond"].shape[0]

        if pointcloud is not None:
            if isinstance(pointcloud, list):
                cond_tensor = torch.tensor(pointcloud).float().cuda().view(-1, 6)
                xyz = cond_tensor[:, :3]
                color_rgb = cond_tensor[:, 3:]
            # Check if point cloud is a numpy array
            elif isinstance(pointcloud, np.ndarray):
                xyz = torch.tensor(pointcloud[:, :3]).float().cuda()
                color_rgb = torch.tensor(pointcloud[:, 3:]).float().cuda()
            else:
                raise ValueError("Invalid point cloud type")

            pointcloud = torch.cat([xyz, color_rgb], dim=-1).unsqueeze(0)
            batch["pc_cond"] = pointcloud

        if "pc_cond" not in batch:
            cond_tokens = self.forward_pdiff_cond(batch)
            sample_iter = self.sampler.sample_batch_progressive(
                batch_size, cond_tokens, device=self.device
            )
            for x in sample_iter:
                samples = x["xstart"]

            denoised_pc = samples.permute(0, 2, 1).float()  # [B, C, N] -> [B, N, C]
            denoised_pc = normalize_pc_bbox(denoised_pc)

            # predict the full 3D conditioned on the denoised point cloud
            batch["pc_cond"] = denoised_pc

        scene_codes, non_postprocessed_codes = self.get_scene_codes(batch)

        # Create a rotation matrix for the final output domain
        rotation = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
        rotation2 = trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
        output_rotation = rotation2 @ rotation

        global_dict = {}
        if self.is_low_vram:
            self._unload_pdiff_modules()
            self._unload_main_modules()
            self._load_estimator_modules()

        if self.image_estimator is not None:
            global_dict.update(
                self.image_estimator(
                    torch.cat([batch["rgb_cond"], batch["mask_cond"]], dim=-1)
                )
            )
        if self.global_estimator is not None and estimate_illumination:
            rotation_torch = (
                torch.tensor(output_rotation)
                .to(self.device, dtype=torch.float32)[:3, :3]
                .unsqueeze(0)
            )
            global_dict.update(
                self.global_estimator(non_postprocessed_codes, rotation=rotation_torch)
            )

        global_dict["pointcloud"] = batch["pc_cond"]

        device = get_device()
        with torch.no_grad():
            with (
                torch.autocast(device_type=device, enabled=False)
                if "cuda" in device
                else nullcontext()
            ):
                meshes = self.triplane_to_meshes(scene_codes)

                rets = []
                for i, mesh in enumerate(meshes):
                    # Check for empty mesh
                    if mesh.v_pos.shape[0] == 0:
                        rets.append(trimesh.Trimesh())
                        continue

                    if remesh == "triangle":
                        mesh = mesh.triangle_remesh(triangle_vertex_count=vertex_count)
                    elif remesh == "quad":
                        mesh = mesh.quad_remesh(quad_vertex_count=vertex_count)
                    else:
                        if vertex_count > 0:
                            print(
                                "Warning: vertex_count is ignored when remesh is none"
                            )

                    if remesh != "none":
                        print(
                            f"After {remesh} remesh the mesh has {mesh.v_pos.shape[0]} verts and {mesh.t_pos_idx.shape[0]} faces",
                        )
                        mesh.unwrap_uv()

                    # Build textures
                    rast = self.baker.rasterize(
                        mesh.v_tex, mesh.t_pos_idx, bake_resolution
                    )
                    bake_mask = self.baker.get_mask(rast)

                    pos_bake = self.baker.interpolate(
                        mesh.v_pos,
                        rast,
                        mesh.t_pos_idx,
                    )
                    gb_pos = pos_bake[bake_mask]

                    tri_query = self.query_triplane(gb_pos, scene_codes[i])[0]
                    decoded = self.decoder(
                        tri_query, exclude=["density", "vertex_offset"]
                    )

                    nrm = self.baker.interpolate(
                        mesh.v_nrm,
                        rast,
                        mesh.t_pos_idx,
                    )
                    gb_nrm = F.normalize(nrm[bake_mask], dim=-1)
                    decoded["normal"] = gb_nrm

                    # Check if any keys in global_dict start with decoded_
                    for k, v in global_dict.items():
                        if k.startswith("decoder_"):
                            decoded[k.replace("decoder_", "")] = v[i]

                    mat_out = {
                        "albedo": decoded["features"],
                        "roughness": decoded["roughness"],
                        "metallic": decoded["metallic"],
                        "normal": normalize(decoded["perturb_normal"]),
                        "bump": None,
                    }

                    for k, v in mat_out.items():
                        if v is None:
                            continue
                        if v.shape[0] == 1:
                            # Skip and directly add a single value
                            mat_out[k] = v[0]
                        else:
                            f = torch.zeros(
                                bake_resolution,
                                bake_resolution,
                                v.shape[-1],
                                dtype=v.dtype,
                                device=v.device,
                            )
                            if v.shape == f.shape:
                                continue
                            if k == "normal":
                                # Use un-normalized tangents here so that larger smaller tris
                                # Don't effect the tangents that much
                                tng = self.baker.interpolate(
                                    mesh.v_tng,
                                    rast,
                                    mesh.t_pos_idx,
                                )
                                gb_tng = tng[bake_mask]
                                gb_tng = F.normalize(gb_tng, dim=-1)
                                gb_btng = F.normalize(
                                    torch.cross(gb_nrm, gb_tng, dim=-1), dim=-1
                                )
                                normal = F.normalize(mat_out["normal"], dim=-1)

                                # Create tangent space matrix and transform normal
                                tangent_matrix = torch.stack(
                                    [gb_tng, gb_btng, gb_nrm], dim=-1
                                )
                                normal_tangent = torch.bmm(
                                    tangent_matrix.transpose(1, 2), normal.unsqueeze(-1)
                                ).squeeze(-1)

                                # Convert from [-1,1] to [0,1] range for storage
                                normal_tangent = (normal_tangent * 0.5 + 0.5).clamp(
                                    0, 1
                                )

                                f[bake_mask] = normal_tangent.view(-1, 3)
                                mat_out["bump"] = f
                            else:
                                f[bake_mask] = v.view(-1, v.shape[-1])
                                mat_out[k] = f

                    def uv_padding(arr):
                        if arr.ndim == 1:
                            return arr
                        return (
                            dilate_fill(
                                arr.permute(2, 0, 1)[None, ...].contiguous(),
                                bake_mask.unsqueeze(0).unsqueeze(0),
                                iterations=bake_resolution // 150,
                            )
                            .squeeze(0)
                            .permute(1, 2, 0)
                            .contiguous()
                        )

                    verts_np = convert_data(mesh.v_pos)
                    faces = convert_data(mesh.t_pos_idx)
                    uvs = convert_data(mesh.v_tex)

                    basecolor_tex = Image.fromarray(
                        float32_to_uint8_np(convert_data(uv_padding(mat_out["albedo"])))
                    ).convert("RGB")
                    basecolor_tex.format = "JPEG"

                    metallic = mat_out["metallic"].squeeze().cpu().item()
                    roughness = mat_out["roughness"].squeeze().cpu().item()

                    if "bump" in mat_out and mat_out["bump"] is not None:
                        bump_np = convert_data(uv_padding(mat_out["bump"]))
                        bump_up = np.ones_like(bump_np)
                        bump_up[..., :2] = 0.5
                        bump_up[..., 2:] = 1
                        bump_tex = Image.fromarray(
                            float32_to_uint8_np(
                                bump_np,
                                dither=True,
                                # Do not dither if something is perfectly flat
                                dither_mask=np.all(
                                    bump_np == bump_up, axis=-1, keepdims=True
                                ).astype(np.float32),
                            )
                        ).convert("RGB")
                        bump_tex.format = (
                            "JPEG"  # PNG would be better but the assets are larger
                        )
                    else:
                        bump_tex = None

                    material = trimesh.visual.material.PBRMaterial(
                        baseColorTexture=basecolor_tex,
                        roughnessFactor=roughness,
                        metallicFactor=metallic,
                        normalTexture=bump_tex,
                    )

                    tmesh = trimesh.Trimesh(
                        vertices=verts_np,
                        faces=faces,
                        visual=trimesh.visual.texture.TextureVisuals(
                            uv=uvs, material=material
                        ),
                    )
                    tmesh.apply_transform(output_rotation)

                    tmesh.invert()

                    rets.append(tmesh)

        return rets, global_dict
