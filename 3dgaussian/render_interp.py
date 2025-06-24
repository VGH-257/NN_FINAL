import torch
import os
import numpy as np
from tqdm import tqdm
from scene import Scene
from gaussian_renderer import render
from gaussian_renderer import GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
import torchvision
from scene.cameras import MiniCam
from argparse import ArgumentParser

def interpolate_views(cam1, cam2, n_frames=30, height=0.0, max_pitch=0.2):
    views = []

    pos1 = cam1.camera_center.cpu().numpy()
    pos2 = cam2.camera_center.cpu().numpy()
    dir1 = -cam1.world_view_transform[:3, 2].cpu().numpy()
    dir2 = -cam2.world_view_transform[:3, 2].cpu().numpy()

    for i in range(n_frames):
        t = i / (n_frames - 1)

        # 插值相机位置
        pos_interp = (1 - t) * pos1 + t * pos2
        pos_interp[1] += height * 4 * t * (1 - t)

        # 插值方向
        dir_interp = (1 - t) * dir1 + t * dir2
        dir_interp = dir_interp / np.linalg.norm(dir_interp)

        # 限制俯仰角：限制方向向量的 Y 分量
        if abs(dir_interp[1]) > max_pitch:
            dir_interp[1] = np.clip(dir_interp[1], -max_pitch, max_pitch)
            dir_interp = dir_interp / np.linalg.norm(dir_interp)

        look_at = pos_interp + dir_interp

        # 构建视图矩阵
        up = np.array([0, 1, 0], dtype=np.float32)
        z_axis = (pos_interp - look_at)
        z_axis /= np.linalg.norm(z_axis)
        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        R = np.stack([x_axis, y_axis, z_axis], axis=1)
        T = -R.T @ pos_interp

        view_matrix = np.eye(4, dtype=np.float32)
        view_matrix[:3, :3] = R.T
        view_matrix[:3, 3] = T

        world_view = torch.tensor(view_matrix.T, dtype=torch.float32).cuda()

        cam = MiniCam(
            width=cam1.image_width,
            height=cam1.image_height,
            fovx=cam1.FoVx,
            fovy=cam1.FoVy,
            znear=cam1.znear,
            zfar=cam1.zfar,
            world_view_transform=world_view,
            full_proj_transform=world_view @ cam1.projection_matrix
        )
        views.append(cam)
    return views



def render_multi_segment_path(scene, gaussians, pipeline, output_folder,
                               view_indices, total_frames, height=0.2):
    cams = scene.getTrainCameras()
    if any(i >= len(cams) for i in view_indices):
        raise ValueError("view_indices 超出训练集长度")

    selected = [cams[i] for i in view_indices]
    all_views = []

    num_segments = len(selected) - 1
    frames_per_segment = [total_frames // num_segments] * num_segments

    # 平滑分配帧数（处理不能整除的情况）
    leftover = total_frames - sum(frames_per_segment)
    for i in range(leftover):
        frames_per_segment[i] += 1

    for i in range(num_segments):
        cam_a = selected[i]
        cam_b = selected[i + 1]
        n_frames = frames_per_segment[i]

        interpolated = interpolate_views(cam_a, cam_b, n_frames=n_frames, height=height)
        all_views.extend(interpolated)

    os.makedirs(output_folder, exist_ok=True)
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    for i, cam in enumerate(tqdm(all_views, desc="Rendering interpolated path")):
        with torch.no_grad():
            out = render(cam, gaussians, pipeline, background)["render"]
            torchvision.utils.save_image(out, os.path.join(output_folder, f"{i:04d}.png"))
            del out
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Multi-view interpolated rendering")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int, help="Iteration to load")
    parser.add_argument("--n_frames", type=int, default=60, help="Total number of interpolated frames to generate")
    parser.add_argument("--view_indices", nargs="+", type=int, required=True,
                        help="Indices of training views to interpolate between, e.g., 0 5 9 15")
    parser.add_argument("--output", default="output/interp_path", type=str, help="Output folder for rendered frames")

    args = get_combined_args(parser)
    safe_state(False)

    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)

    render_multi_segment_path(
        scene=scene,
        gaussians=gaussians,
        pipeline=pipeline.extract(args),
        output_folder=args.output,
        view_indices=args.view_indices,
        total_frames=args.n_frames
    )
