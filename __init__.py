import base64
import logging
import os
import random
import sys
from contextlib import nullcontext

import comfy.model_management
import folder_paths
import numpy as np
import torch
import trimesh
from PIL import Image
from trimesh.exchange import gltf

sys.path.append(os.path.dirname(__file__))
from spar3d.models.mesh import QUAD_REMESH_AVAILABLE, TRIANGLE_REMESH_AVAILABLE
from spar3d.system import SPAR3D
from spar3d.utils import foreground_crop

SPAR3D_CATEGORY = "SPAR3D"
SPAR3D_MODEL_NAME = "stabilityai/spar3d"


class SPAR3DLoader:
    CATEGORY = SPAR3D_CATEGORY
    FUNCTION = "load"
    RETURN_NAMES = ("spar3d_model",)
    RETURN_TYPES = ("SPAR3D_MODEL",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "low_vram_mode": ("BOOLEAN", {"default": False}),
            }
        }

    def load(self, low_vram_mode=False):
        device = comfy.model_management.get_torch_device()
        model = SPAR3D.from_pretrained(
            SPAR3D_MODEL_NAME,
            config_name="config.yaml",
            weight_name="model.safetensors",
            low_vram_mode=low_vram_mode,
        )
        model.to(device)
        model.eval()

        return (model,)


class SPAR3DPreview:
    CATEGORY = SPAR3D_CATEGORY
    FUNCTION = "preview"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mesh": ("MESH",)}}

    def preview(self, mesh):
        glbs = []
        for m in mesh:
            scene = trimesh.Scene(m)
            glb_data = gltf.export_glb(scene, include_normals=True)
            glb_base64 = base64.b64encode(glb_data).decode("utf-8")
            glbs.append(glb_base64)
        return {"ui": {"glbs": glbs}}


class SPAR3DSampler:
    CATEGORY = SPAR3D_CATEGORY
    FUNCTION = "predict"
    RETURN_NAMES = ("mesh", "pointcloud")
    RETURN_TYPES = ("MESH", "POINTCLOUD")

    @classmethod
    def INPUT_TYPES(s):
        remesh_choices = ["none"]
        if TRIANGLE_REMESH_AVAILABLE:
            remesh_choices.append("triangle")
        if QUAD_REMESH_AVAILABLE:
            remesh_choices.append("quad")

        opt_dict = {
            "mask": ("MASK",),
            "pointcloud": ("POINTCLOUD",),
            "target_type": (["none", "vertex", "face"],),
            "target_count": (
                "INT",
                {"default": 1000, "min": 3, "max": 20000, "step": 1},
            ),
            "guidance_scale": (
                "FLOAT",
                {"default": 3.0, "min": 1.0, "max": 5.0, "step": 0.05},
            ),
            "seed": (
                "INT",
                {"default": 42, "min": 0, "max": 2**32 - 1, "step": 1},
            ),
        }
        if TRIANGLE_REMESH_AVAILABLE or QUAD_REMESH_AVAILABLE:
            opt_dict["remesh"] = (remesh_choices,)

        return {
            "required": {
                "model": ("SPAR3D_MODEL",),
                "image": ("IMAGE",),
                "foreground_ratio": (
                    "FLOAT",
                    {"default": 1.3, "min": 1.0, "max": 2.0, "step": 0.01},
                ),
                "texture_resolution": (
                    "INT",
                    {"default": 1024, "min": 512, "max": 2048, "step": 256},
                ),
            },
            "optional": opt_dict,
        }

    def predict(
        s,
        model,
        image,
        mask,
        foreground_ratio,
        texture_resolution,
        pointcloud=None,
        remesh="none",
        target_type="none",
        target_count=1000,
        guidance_scale=3.0,
        seed=42,
    ):
        if image.shape[0] != 1:
            raise ValueError("Only one image can be processed at a time")

        vertex_count = (
            -1
            if target_type == "none"
            else (target_count // 2 if target_type == "face" else target_count)
        )

        pil_image = Image.fromarray(
            torch.clamp(torch.round(255.0 * image[0]), 0, 255)
            .type(torch.uint8)
            .cpu()
            .numpy()
        )

        if mask is not None:
            print("Using Mask")
            mask_np = np.clip(255.0 * mask[0].detach().cpu().numpy(), 0, 255).astype(
                np.uint8
            )
            mask_pil = Image.fromarray(mask_np, mode="L")
            pil_image.putalpha(mask_pil)
        else:
            if image.shape[3] != 4:
                print("No mask or alpha channel detected, Converting to RGBA")
                pil_image = pil_image.convert("RGBA")

        pil_image = foreground_crop(pil_image, foreground_ratio)

        model.cfg.guidance_scale = guidance_scale
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        print(remesh)
        with torch.no_grad():
            with (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if "cuda" in comfy.model_management.get_torch_device().type
                else nullcontext()
            ):
                if not TRIANGLE_REMESH_AVAILABLE and remesh == "triangle":
                    raise ImportError(
                        "Triangle remeshing requires gpytoolbox to be installed"
                    )
                if not QUAD_REMESH_AVAILABLE and remesh == "quad":
                    raise ImportError("Quad remeshing requires pynim to be installed")
                mesh, glob_dict = model.run_image(
                    pil_image,
                    bake_resolution=texture_resolution,
                    pointcloud=pointcloud,
                    remesh=remesh,
                    vertex_count=vertex_count,
                )

        if mesh.vertices.shape[0] == 0:
            raise ValueError("No subject detected in the image")

        return (
            [mesh],
            glob_dict["pointcloud"].view(-1).detach().cpu().numpy().tolist(),
        )


class SPAR3DSave:
    CATEGORY = SPAR3D_CATEGORY
    FUNCTION = "save"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",),
                "filename_prefix": ("STRING", {"default": "SPAR3D"}),
            }
        }

    def __init__(self):
        self.type = "output"

    def save(self, mesh, filename_prefix):
        output_dir = folder_paths.get_output_directory()
        glbs = []
        for idx, m in enumerate(mesh):
            scene = trimesh.Scene(m)
            glb_data = gltf.export_glb(scene, include_normals=True)
            logging.info(f"Generated GLB model with {len(glb_data)} bytes")

            full_output_folder, filename, counter, subfolder, filename_prefix = (
                folder_paths.get_save_image_path(filename_prefix, output_dir)
            )
            filename = filename.replace("%batch_num%", str(idx))
            out_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.glb")
            with open(out_path, "wb") as f:
                f.write(glb_data)
            glbs.append(base64.b64encode(glb_data).decode("utf-8"))
        return {"ui": {"glbs": glbs}}


class SPAR3DPointCloudLoader:
    CATEGORY = SPAR3D_CATEGORY
    FUNCTION = "load_pointcloud"
    RETURN_TYPES = ("POINTCLOUD",)
    RETURN_NAMES = ("pointcloud",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": ("STRING", {"default": None}),
            }
        }

    def load_pointcloud(self, file):
        if file is None or file == "":
            return (None,)
        # Load the mesh using trimesh
        mesh = trimesh.load(file)

        # Extract vertices and colors
        vertices = mesh.vertices

        # Get vertex colors, defaulting to white if none exist
        if mesh.visual.vertex_colors is not None:
            colors = (
                mesh.visual.vertex_colors[:, :3] / 255.0
            )  # Convert 0-255 to 0-1 range
        else:
            colors = np.ones((len(vertices), 3))

        # Interleave XYZ and RGB values
        point_cloud = []
        for vertex, color in zip(vertices, colors):
            point_cloud.extend(
                [
                    float(vertex[0]),
                    float(vertex[1]),
                    float(vertex[2]),
                    float(color[0]),
                    float(color[1]),
                    float(color[2]),
                ]
            )

        return (point_cloud,)


class SPAR3DPointCloudSaver:
    CATEGORY = SPAR3D_CATEGORY
    FUNCTION = "save_pointcloud"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pointcloud": ("POINTCLOUD",),
                "filename_prefix": ("STRING", {"default": "SPAR3D"}),
            }
        }

    def save_pointcloud(self, pointcloud, filename_prefix):
        if pointcloud is None:
            return {"ui": {"text": "No point cloud data to save"}}

        # Reshape the flat list into points with XYZ and RGB
        points = np.array(pointcloud).reshape(-1, 6)

        # Create vertex array for PLY
        vertex_array = np.zeros(
            len(points),
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ],
        )

        # Fill vertex array
        vertex_array["x"] = points[:, 0]
        vertex_array["y"] = points[:, 1]
        vertex_array["z"] = points[:, 2]
        # Convert RGB from 0-1 to 0-255 range
        vertex_array["red"] = (points[:, 3] * 255).astype(np.uint8)
        vertex_array["green"] = (points[:, 4] * 255).astype(np.uint8)
        vertex_array["blue"] = (points[:, 5] * 255).astype(np.uint8)

        # Create PLY object
        ply_data = trimesh.PointCloud(
            vertices=points[:, :3], colors=points[:, 3:] * 255
        )

        # Save to file
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(filename_prefix, output_dir)
        )
        out_path = os.path.join(full_output_folder, f"{filename}_{counter:05}.ply")

        ply_data.export(out_path)

        return {"ui": {"text": f"Saved point cloud to {out_path}"}}


NODE_DISPLAY_NAME_MAPPINGS = {
    "SPAR3DLoader": "SPAR3D Loader",
    "SPAR3DPreview": "SPAR3D Preview",
    "SPAR3DSampler": "SPAR3D Sampler",
    "SPAR3DSave": "SPAR3D Save",
    "SPAR3DPointCloudLoader": "SPAR3D Point Cloud Loader",
    "SPAR3DPointCloudSaver": "SPAR3D Point Cloud Saver",
}

NODE_CLASS_MAPPINGS = {
    "SPAR3DLoader": SPAR3DLoader,
    "SPAR3DPreview": SPAR3DPreview,
    "SPAR3DSampler": SPAR3DSampler,
    "SPAR3DSave": SPAR3DSave,
    "SPAR3DPointCloudLoader": SPAR3DPointCloudLoader,
    "SPAR3DPointCloudSaver": SPAR3DPointCloudSaver,
}

WEB_DIRECTORY = "./comfyui"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
