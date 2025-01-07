import os
import random
import tempfile
import time
import zipfile
from contextlib import nullcontext
from functools import lru_cache
from typing import Any

import cv2
import gradio as gr
import numpy as np
import torch
import trimesh
from gradio_litmodel3d import LitModel3D
from gradio_pointcloudeditor import PointCloudEditor
from PIL import Image
from transparent_background import Remover

import spar3d.utils as spar3d_utils
from spar3d.models.mesh import QUAD_REMESH_AVAILABLE, TRIANGLE_REMESH_AVAILABLE
from spar3d.system import SPAR3D

os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.environ.get("TMPDIR", "/tmp"), "gradio")

bg_remover = Remover()  # default setting

COND_WIDTH = 512
COND_HEIGHT = 512
COND_DISTANCE = 2.2
COND_FOVY = 0.591627
BACKGROUND_COLOR = [0.5, 0.5, 0.5]

# Cached. Doesn't change
c2w_cond = spar3d_utils.default_cond_c2w(COND_DISTANCE)
intrinsic, intrinsic_normed_cond = spar3d_utils.create_intrinsic_from_fov_rad(
    COND_FOVY, COND_HEIGHT, COND_WIDTH
)

generated_files = []

# Delete previous gradio temp dir folder
if os.path.exists(os.environ["GRADIO_TEMP_DIR"]):
    print(f"Deleting {os.environ['GRADIO_TEMP_DIR']}")
    import shutil

    shutil.rmtree(os.environ["GRADIO_TEMP_DIR"])

device = spar3d_utils.get_device()

model = SPAR3D.from_pretrained(
    "stabilityai/stable-point-aware-3d",
    config_name="config.yaml",
    weight_name="model.safetensors",
)
model.eval()
model = model.to(device)

example_files = [
    os.path.join("demo_files/examples", f) for f in os.listdir("demo_files/examples")
]


def create_zip_file(glb_file, pc_file, illumination_file):
    if not all([glb_file, pc_file, illumination_file]):
        return None

    # Create a temporary zip file
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "spar3d_output.zip")

    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(glb_file, "mesh.glb")
        zipf.write(pc_file, "points.ply")
        zipf.write(illumination_file, "illumination.hdr")

    generated_files.append(zip_path)
    return zip_path


def forward_model(
    batch,
    system,
    guidance_scale=3.0,
    seed=0,
    device="cuda",
    remesh_option="none",
    vertex_count=-1,
    texture_resolution=1024,
):
    batch_size = batch["rgb_cond"].shape[0]

    # prepare the condition for point cloud generation
    # set seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cond_tokens = system.forward_pdiff_cond(batch)

    if "pc_cond" not in batch:
        sample_iter = system.sampler.sample_batch_progressive(
            batch_size,
            cond_tokens,
            guidance_scale=guidance_scale,
            device=device,
        )
        for x in sample_iter:
            samples = x["xstart"]
        batch["pc_cond"] = samples.permute(0, 2, 1).float()
        batch["pc_cond"] = spar3d_utils.normalize_pc_bbox(batch["pc_cond"])

    # subsample to the 512 points
    batch["pc_cond"] = batch["pc_cond"][
        :, torch.randperm(batch["pc_cond"].shape[1])[:512]
    ]

    # get the point cloud
    xyz = batch["pc_cond"][0, :, :3].cpu().numpy()
    color_rgb = (batch["pc_cond"][0, :, 3:6] * 255).cpu().numpy().astype(np.uint8)
    pc_rgb_trimesh = trimesh.PointCloud(vertices=xyz, colors=color_rgb)

    # forward for the final mesh
    trimesh_mesh, _glob_dict = model.generate_mesh(
        batch,
        texture_resolution,
        remesh=remesh_option,
        vertex_count=vertex_count,
        estimate_illumination=True,
    )
    trimesh_mesh = trimesh_mesh[0]
    illumination = _glob_dict["illumination"]

    return trimesh_mesh, pc_rgb_trimesh, illumination.cpu().detach().numpy()[0]


def run_model(
    input_image,
    guidance_scale,
    random_seed,
    pc_cond,
    remesh_option,
    vertex_count,
    texture_resolution,
):
    start = time.time()
    with torch.no_grad():
        with (
            torch.autocast(device_type=device, dtype=torch.bfloat16)
            if "cuda" in device
            else nullcontext()
        ):
            model_batch = create_batch(input_image)
            model_batch = {k: v.to(device) for k, v in model_batch.items()}

            if pc_cond is not None:
                # Check if pc_cond is a list
                if isinstance(pc_cond, list):
                    cond_tensor = torch.tensor(pc_cond).float().cuda().view(-1, 6)
                    xyz = cond_tensor[:, :3]
                    color_rgb = cond_tensor[:, 3:]
                elif isinstance(pc_cond, dict):
                    xyz = torch.tensor(pc_cond["positions"]).float().cuda()
                    color_rgb = torch.tensor(pc_cond["colors"]).float().cuda()
                else:
                    xyz = torch.tensor(pc_cond.vertices).float().cuda()
                    color_rgb = (
                        torch.tensor(pc_cond.colors[:, :3]).float().cuda() / 255.0
                    )
                model_batch["pc_cond"] = torch.cat([xyz, color_rgb], dim=-1).unsqueeze(
                    0
                )
                # sub-sample the point cloud to the target number of points
                if model_batch["pc_cond"].shape[1] > 512:
                    idx = torch.randperm(model_batch["pc_cond"].shape[1])[:512]
                    model_batch["pc_cond"] = model_batch["pc_cond"][:, idx]
                elif model_batch["pc_cond"].shape[1] < 512:
                    num_points = model_batch["pc_cond"].shape[1]
                    gr.Warning(
                        f"The uploaded point cloud should have at least 512 points. This point cloud only has {num_points}. Results may be worse."
                    )
                    pad = 512 - num_points
                    sampled_idx = torch.randint(
                        0, model_batch["pc_cond"].shape[1], (pad,)
                    )
                    model_batch["pc_cond"] = torch.cat(
                        [
                            model_batch["pc_cond"],
                            model_batch["pc_cond"][:, sampled_idx],
                        ],
                        dim=1,
                    )

            trimesh_mesh, trimesh_pc, illumination_map = forward_model(
                model_batch,
                model,
                guidance_scale=guidance_scale,
                seed=random_seed,
                device="cuda",
                remesh_option=remesh_option.lower(),
                vertex_count=vertex_count,
                texture_resolution=texture_resolution,
            )

    # Create new tmp file
    temp_dir = tempfile.mkdtemp()
    tmp_file = os.path.join(temp_dir, "mesh.glb")

    trimesh_mesh.export(tmp_file, file_type="glb", include_normals=True)
    generated_files.append(tmp_file)

    tmp_file_pc = os.path.join(temp_dir, "points.ply")
    trimesh_pc.export(tmp_file_pc)
    generated_files.append(tmp_file_pc)

    tmp_file_illumination = os.path.join(temp_dir, "illumination.hdr")
    cv2.imwrite(tmp_file_illumination, illumination_map)
    generated_files.append(tmp_file_illumination)

    print("Generation took:", time.time() - start, "s")

    return tmp_file, tmp_file_pc, tmp_file_illumination, trimesh_pc


def create_batch(input_image: Image) -> dict[str, Any]:
    img_cond = (
        torch.from_numpy(
            np.asarray(input_image.resize((COND_WIDTH, COND_HEIGHT))).astype(np.float32)
            / 255.0
        )
        .float()
        .clip(0, 1)
    )
    mask_cond = img_cond[:, :, -1:]
    rgb_cond = torch.lerp(
        torch.tensor(BACKGROUND_COLOR)[None, None, :], img_cond[:, :, :3], mask_cond
    )

    batch_elem = {
        "rgb_cond": rgb_cond,
        "mask_cond": mask_cond,
        "c2w_cond": c2w_cond.unsqueeze(0),
        "intrinsic_cond": intrinsic.unsqueeze(0),
        "intrinsic_normed_cond": intrinsic_normed_cond.unsqueeze(0),
    }
    # Add batch dim
    batched = {k: v.unsqueeze(0) for k, v in batch_elem.items()}
    return batched


@lru_cache
def checkerboard(squares: int, size: int, min_value: float = 0.5):
    base = np.zeros((squares, squares)) + min_value
    base[1::2, ::2] = 1
    base[::2, 1::2] = 1

    repeat_mult = size // squares
    return (
        base.repeat(repeat_mult, axis=0)
        .repeat(repeat_mult, axis=1)[:, :, None]
        .repeat(3, axis=-1)
    )


def remove_background(input_image: Image) -> Image:
    return bg_remover.process(input_image.convert("RGB"))


def show_mask_img(input_image: Image) -> Image:
    img_numpy = np.array(input_image)
    alpha = img_numpy[:, :, 3] / 255.0
    chkb = checkerboard(32, 512) * 255
    new_img = img_numpy[..., :3] * alpha[:, :, None] + chkb * (1 - alpha[:, :, None])
    return Image.fromarray(new_img.astype(np.uint8), mode="RGB")


def process_model_run(
    background_state,
    guidance_scale,
    random_seed,
    pc_cond,
    remesh_option,
    vertex_count_type,
    vertex_count,
    texture_resolution,
):
    # Adjust vertex count based on selection
    final_vertex_count = (
        -1
        if vertex_count_type == "Keep Vertex Count"
        else (
            vertex_count // 2
            if vertex_count_type == "Target Face Count"
            else vertex_count
        )
    )
    print(
        f"Final vertex count: {final_vertex_count} with type {vertex_count_type} and vertex count {vertex_count}"
    )

    glb_file, pc_file, illumination_file, pc_plot = run_model(
        background_state,
        guidance_scale,
        random_seed,
        pc_cond,
        remesh_option,
        final_vertex_count,
        texture_resolution,
    )
    # Create a single float list of x y z r g b
    point_list = []
    for i in range(pc_plot.vertices.shape[0]):
        point_list.extend(
            [
                pc_plot.vertices[i, 0],
                pc_plot.vertices[i, 1],
                pc_plot.vertices[i, 2],
                pc_plot.colors[i, 0] / 255.0,
                pc_plot.colors[i, 1] / 255.0,
                pc_plot.colors[i, 2] / 255.0,
            ]
        )

    return glb_file, pc_file, illumination_file, point_list


def regenerate_run(
    background_state,
    guidance_scale,
    random_seed,
    pc_cond,
    remesh_option,
    vertex_count_type,
    vertex_count,
    texture_resolution,
):
    glb_file, pc_file, illumination_file, point_list = process_model_run(
        background_state,
        guidance_scale,
        random_seed,
        pc_cond,
        remesh_option,
        vertex_count_type,
        vertex_count,
        texture_resolution,
    )
    zip_file = create_zip_file(glb_file, pc_file, illumination_file)

    return (
        gr.update(),  # run_btn
        gr.update(),  # img_proc_state
        gr.update(),  # background_remove_state
        gr.update(),  # preview_removal
        gr.update(value=glb_file, visible=True),  # output_3d
        gr.update(visible=True),  # hdr_row
        illumination_file,  # hdr_file
        gr.update(visible=True),  # point_cloud_row
        gr.update(value=point_list),  # point_cloud_editor
        gr.update(value=pc_file),  # pc_download
        gr.update(visible=False),  # regenerate_btn
        gr.update(value=zip_file, visible=True),  # download_all_btn
    )


def run_button(
    run_btn,
    input_image,
    background_state,
    foreground_ratio,
    no_crop,
    guidance_scale,
    random_seed,
    pc_upload,
    pc_cond_file,
    remesh_option,
    vertex_count_type,
    vertex_count,
    texture_resolution,
):
    if run_btn == "Run":
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if pc_upload:
            # make sure the pc_cond_file has been uploaded
            try:
                pc_cond = trimesh.load(pc_cond_file.name)
            except Exception:
                raise gr.Error(
                    "Please upload a valid point cloud ply file as condition."
                )
        else:
            pc_cond = None

        glb_file, pc_file, illumination_file, pc_list = process_model_run(
            background_state,
            guidance_scale,
            random_seed,
            pc_cond,
            remesh_option,
            vertex_count_type,
            vertex_count,
            texture_resolution,
        )

        zip_file = create_zip_file(glb_file, pc_file, illumination_file)

        if torch.cuda.is_available():
            print("Peak Memory:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")
        elif torch.backends.mps.is_available():
            print(
                "Peak Memory:", torch.mps.driver_allocated_memory() / 1024 / 1024, "MB"
            )

        return (
            gr.update(),  # run_btn
            gr.update(),  # img_proc_state
            gr.update(),  # background_remove_state
            gr.update(),  # preview_removal
            gr.update(value=glb_file, visible=True),  # output_3d
            gr.update(visible=True),  # hdr_row
            illumination_file,  # hdr_file
            gr.update(visible=True),  # point_cloud_row
            gr.update(value=pc_list),  # point_cloud_editor
            gr.update(value=pc_file),  # pc_download
            gr.update(visible=False),  # regenerate_btn
            gr.update(value=zip_file, visible=True),  # download_all_btn
        )

    elif run_btn == "Remove Background":
        rem_removed = remove_background(input_image)

        fr_res = spar3d_utils.foreground_crop(
            rem_removed,
            crop_ratio=foreground_ratio,
            newsize=(COND_WIDTH, COND_HEIGHT),
            no_crop=no_crop,
        )

        return (
            gr.update(value="Run", visible=True),  # run_btn
            rem_removed,  # img_proc_state,
            fr_res,  # background_remove_state
            gr.update(value=show_mask_img(fr_res), visible=True),  # preview_removal
            gr.update(value=None, visible=False),  # output_3d
            gr.update(visible=False),  # hdr_row
            None,  # hdr_file
            gr.update(visible=False),  # point_cloud_row
            gr.update(value=None),  # point_cloud_editor
            gr.update(value=None),  # pc_download
            gr.update(visible=False),  # regenerate_btn
            gr.update(value=None, visible=False),  # download_all_btn
        )


def requires_bg_remove(image, fr, no_crop):
    if image is None:
        return (
            gr.update(visible=False, value="Run"),  # run_Btn
            None,  # img_proc_state
            None,  # background_remove_state
            gr.update(value=None, visible=False),  # preview_removal
            gr.update(value=None, visible=False),  # output_3d
            gr.update(value=None, visible=False),  # hdr_row
            None,  # hdr_file
            gr.update(visible=False),  # point_cloud_row
            gr.update(value=None),  # point_cloud_editor
            gr.update(value=None),  # pc_download
            gr.update(visible=False),  # regenerate_btn
            gr.update(value=None, visible=False),  # download_all_btn
        )
    alpha_channel = np.array(image.getchannel("A"))
    min_alpha = alpha_channel.min()

    if min_alpha == 0:
        print("Already has alpha")
        fr_res = spar3d_utils.foreground_crop(
            image, fr, newsize=(COND_WIDTH, COND_HEIGHT), no_crop=no_crop
        )
        return (
            gr.update(value="Run", visible=True),  # run_Btn
            image,  # img_proc_state
            fr_res,  # background_remove_state
            gr.update(value=show_mask_img(fr_res), visible=True),  # preview_removal
            gr.update(value=None, visible=False),  # output_3d
            gr.update(visible=False),  # hdr_row
            None,  # hdr_file
            gr.update(visible=False),  # point_cloud_row
            gr.update(value=None),  # point_cloud_editor
            gr.update(value=None),  # pc_download
            gr.update(visible=False),  # regenerate_btn
            gr.update(value=None, visible=False),  # download_all_btn
        )
    return (
        gr.update(value="Remove Background", visible=True),  # run_Btn
        None,  # img_proc_state
        None,  # background_remove_state
        gr.update(value=None, visible=False),  # preview_removal
        gr.update(value=None, visible=False),  # output_3d
        gr.update(visible=False),  # hdr_row
        None,  # hdr_file
        gr.update(visible=False),  # point_cloud_row
        gr.update(value=None),  # point_cloud_editor
        gr.update(value=None),  # pc_download
        gr.update(visible=False),  # regenerate_btn
        gr.update(value=None, visible=False),  # download_all_btn
    )


def update_foreground_ratio(img_proc, fr, no_crop):
    foreground_res = spar3d_utils.foreground_crop(
        img_proc, fr, newsize=(COND_WIDTH, COND_HEIGHT), no_crop=no_crop
    )
    return (
        foreground_res,
        gr.update(value=show_mask_img(foreground_res)),
    )


def update_resolution_controls(remesh_choice, vertex_count_type):
    show_controls = remesh_choice.lower() != "none"
    show_vertex_count = vertex_count_type != "Keep Vertex Count"
    return (
        gr.update(visible=show_controls),  # vertex_count_type
        gr.update(visible=show_controls and show_vertex_count),  # vertex_count_slider
    )


with gr.Blocks() as demo:
    img_proc_state = gr.State()
    background_remove_state = gr.State()
    hdr_illumination_file_state = gr.State()
    gr.Markdown(
        """
    # SPAR3D: Stable Point-Aware Reconstruction of 3D Objects from Single Images

    SPAR3D is a state-of-the-art method for 3D mesh reconstruction from a single image. This demo allows you to upload an image and generate a 3D mesh model from it. A feature of SPAR3D is it generates point clouds as intermediate representation before producing the mesh. You can edit the point cloud to adjust the final mesh. We provide a simple point cloud editor in this demo, where you can drag, recolor and rescale the point clouds. If you have more advanced editing needs (e.g. box selection, duplication, local streching, etc.), you can download the point cloud and edit it in softwares such as MeshLab or Blender. The edited point cloud can then be uploaded to this demo to generate a new 3D model by checking the "Point cloud upload" box.

    **Tips**

    1. If the image does not have a valid alpha channel, it will go through the background removal step. Our built-in background removal can be inaccurate sometimes, which will result in poor mesh quality. In such cases, you can use external background removal tools to obtain a RGBA image before uploading here.
    2. You can adjust the foreground ratio to control the size of the foreground object. This may have major impact on the final mesh.
    3. Guidance scale controls the strength of the image condition in the point cloud generation process. A higher value may result in higher mesh fidelity, but the variability by changing the random seed will be lower. Note that the guidance scale and the seed are not effective when the point cloud is manually uploaded.
    4. Our online editor supports multi-selection by holding down the shift key. This allows you to recolor multiple points at once.
    5. The editing should mainly alter the unseen parts of the object. Visible parts can be edited, but the edits should be consistent with the image. Editing the visible parts in a way that contradicts the image may result in poor mesh quality.
    6. You can upload your own HDR environment map to light the 3D model.
    """
    )
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                input_img = gr.Image(
                    type="pil", label="Input Image", sources="upload", image_mode="RGBA"
                )
                preview_removal = gr.Image(
                    label="Preview Background Removal",
                    type="pil",
                    image_mode="RGB",
                    interactive=False,
                    visible=False,
                )

            gr.Markdown("### Input Controls")
            with gr.Group():
                with gr.Row():
                    no_crop = gr.Checkbox(label="No cropping", value=False)
                    pc_upload = gr.Checkbox(label="Point cloud upload", value=False)

                pc_cond_file = gr.File(
                    label="Point Cloud Upload",
                    file_types=[".ply"],
                    file_count="single",
                    visible=False,
                )

                foreground_ratio = gr.Slider(
                    label="Padding Ratio",
                    minimum=1.0,
                    maximum=2.0,
                    value=1.3,
                    step=0.05,
                )

            pc_upload.change(
                lambda x: gr.update(visible=x),
                inputs=pc_upload,
                outputs=[pc_cond_file],
            )

            no_crop.change(
                update_foreground_ratio,
                inputs=[img_proc_state, foreground_ratio, no_crop],
                outputs=[background_remove_state, preview_removal],
            )

            foreground_ratio.change(
                update_foreground_ratio,
                inputs=[img_proc_state, foreground_ratio, no_crop],
                outputs=[background_remove_state, preview_removal],
            )

            gr.Markdown("### Point Diffusion Controls")
            with gr.Group():
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1.0,
                    maximum=10.0,
                    value=3.0,
                    step=1.0,
                )

                random_seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=10000,
                    value=0,
                    step=1,
                )

            no_remesh = not TRIANGLE_REMESH_AVAILABLE and not QUAD_REMESH_AVAILABLE
            gr.Markdown(
                "### Texture Controls"
                if no_remesh
                else "### Meshing and Texture Controls"
            )
            with gr.Group():
                remesh_choices = ["None"]
                if TRIANGLE_REMESH_AVAILABLE:
                    remesh_choices.append("Triangle")
                if QUAD_REMESH_AVAILABLE:
                    remesh_choices.append("Quad")

                remesh_option = gr.Radio(
                    choices=remesh_choices,
                    label="Remeshing",
                    value="None",
                    visible=not no_remesh,
                )

                vertex_count_type = gr.Radio(
                    choices=[
                        "Keep Vertex Count",
                        "Target Vertex Count",
                        "Target Face Count",
                    ],
                    label="Mesh Resolution Control",
                    value="Keep Vertex Count",
                    visible=False,
                )

                vertex_count_slider = gr.Slider(
                    label="Target Count",
                    minimum=0,
                    maximum=20000,
                    value=2000,
                    visible=False,
                )

                texture_size = gr.Slider(
                    label="Texture Size",
                    minimum=512,
                    maximum=2048,
                    value=1024,
                    step=256,
                    visible=True,
                )

            remesh_option.change(
                update_resolution_controls,
                inputs=[remesh_option, vertex_count_type],
                outputs=[vertex_count_type, vertex_count_slider],
            )

            vertex_count_type.change(
                update_resolution_controls,
                inputs=[remesh_option, vertex_count_type],
                outputs=[vertex_count_type, vertex_count_slider],
            )

            run_btn = gr.Button("Run", variant="primary", visible=False)

        with gr.Column():
            with gr.Group(visible=False) as point_cloud_row:
                point_size_slider = gr.Slider(
                    label="Point Size",
                    minimum=0.01,
                    maximum=1.0,
                    value=0.2,
                    step=0.01,
                )
                point_cloud_editor = PointCloudEditor(
                    up_axis="Z",
                    forward_axis="X",
                    lock_scale_z=True,
                    lock_scale_y=True,
                    visible=True,
                )

                pc_download = gr.File(
                    label="Point Cloud Download",
                    file_types=[".ply"],
                    file_count="single",
                )
            point_size_slider.change(
                fn=lambda x: gr.update(point_size=x),
                inputs=point_size_slider,
                outputs=point_cloud_editor,
            )

            regenerate_btn = gr.Button(
                "Re-run with point cloud", variant="primary", visible=False
            )

            output_3d = LitModel3D(
                label="3D Model",
                visible=False,
                clear_color=[0.0, 0.0, 0.0, 0.0],
                tonemapping="aces",
                contrast=1.0,
                scale=1.0,
            )
            with gr.Column(visible=False, scale=1.0) as hdr_row:
                gr.Markdown(
                    """## HDR Environment Map

                Select an HDR environment map to light the 3D model. You can also upload your own HDR environment maps.
                """
                )

                with gr.Row():
                    hdr_illumination_file = gr.File(
                        label="HDR Env Map",
                        file_types=[".hdr"],
                        file_count="single",
                    )
                    example_hdris = [
                        os.path.join("demo_files/hdri", f)
                        for f in os.listdir("demo_files/hdri")
                    ]
                    hdr_illumination_example = gr.Examples(
                        examples=example_hdris,
                        inputs=hdr_illumination_file,
                    )

                    def update_hdr_illumination_file(state, cur_update):
                        # If the current value of hdr_illumination_file is the same as cur_update, then we don't need to update
                        if (
                            hdr_illumination_file.value is not None
                            and hdr_illumination_file.value == cur_update
                        ):
                            return (
                                gr.update(),
                                gr.update(),
                            )
                        update_value = cur_update if cur_update is not None else state
                        if update_value is not None:
                            return (
                                gr.update(value=update_value),
                                gr.update(
                                    env_map=(
                                        update_value.name
                                        if isinstance(update_value, gr.File)
                                        else update_value
                                    )
                                ),
                            )
                        return (gr.update(value=None), gr.update(env_map=None))

                    hdr_illumination_file.change(
                        update_hdr_illumination_file,
                        inputs=[hdr_illumination_file_state, hdr_illumination_file],
                        outputs=[hdr_illumination_file, output_3d],
                    )

            download_all_btn = gr.File(
                label="Download All Files (ZIP)", file_count="single", visible=False
            )

    hdr_illumination_file_state.change(
        fn=lambda x: gr.update(value=x),
        inputs=hdr_illumination_file_state,
        outputs=hdr_illumination_file,
    )

    examples = gr.Examples(
        examples=example_files, inputs=input_img, examples_per_page=11
    )

    input_img.change(
        requires_bg_remove,
        inputs=[input_img, foreground_ratio, no_crop],
        outputs=[
            run_btn,
            img_proc_state,
            background_remove_state,
            preview_removal,
            output_3d,
            hdr_row,
            hdr_illumination_file_state,
            point_cloud_row,
            point_cloud_editor,
            pc_download,
            regenerate_btn,
            download_all_btn,
        ],
    )

    point_cloud_editor.edit(
        fn=lambda _x: gr.update(visible=True),
        inputs=point_cloud_editor,
        outputs=regenerate_btn,
    )

    regenerate_btn.click(
        regenerate_run,
        inputs=[
            background_remove_state,
            guidance_scale,
            random_seed,
            point_cloud_editor,
            remesh_option,
            vertex_count_type,
            vertex_count_slider,
            texture_size,
        ],
        outputs=[
            run_btn,
            img_proc_state,
            background_remove_state,
            preview_removal,
            output_3d,
            hdr_row,
            hdr_illumination_file_state,
            point_cloud_row,
            point_cloud_editor,
            pc_download,
            regenerate_btn,
            download_all_btn,
        ],
    )

    run_btn.click(
        run_button,
        inputs=[
            run_btn,
            input_img,
            background_remove_state,
            foreground_ratio,
            no_crop,
            guidance_scale,
            random_seed,
            pc_upload,
            pc_cond_file,
            remesh_option,
            vertex_count_type,
            vertex_count_slider,
            texture_size,
        ],
        outputs=[
            run_btn,
            img_proc_state,
            background_remove_state,
            preview_removal,
            output_3d,
            hdr_row,
            hdr_illumination_file_state,
            point_cloud_row,
            point_cloud_editor,
            pc_download,
            regenerate_btn,
            download_all_btn,
        ],
    )

demo.queue().launch(share=False)
