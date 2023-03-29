from pathlib import Path
import argparse
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
from testbed import Testbed
from manifest import Frame, Manifest
import time
from tqdm import tqdm
import shutil
import json
from PIL import Image
import gc
import pandas as pd
from common import *
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ThreadPoolExecutor
import copy
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run neural graphics primitives testbed with additional configuration & output options")
    parser.add_argument("--scene", "--training_data", required=True,
                        help="The scene to load. Can be the scene's name or a full path to the training data.")

    parser.add_argument("--test_scene", default="", help="The scene to test on")

    parser.add_argument("--config", default="base",
                        help="Name of config file to use")

    parser.add_argument("--degrees_per_keyframe", default=2,
                        help="How many degrees to rotate per keyframe")

    parser.add_argument("--translation_per_keyframe", type=float, default=0.01,
                        help="How many meters to translate per keyframe")

    parser.add_argument("--max_translation", default=2.0, type=float,
                        help="Distance to travel before reversing")

    parser.add_argument("--gui", action="store_true",
                        help="Run the testbed GUI interactively.")

    parser.add_argument("--save", action="store_true", help="Save the dataset")

    parser.add_argument("--num_keyframes", default=0, type=int,
                        help="Number of keyframes to advance animation")

    parser.add_argument("--static_train_steps", default=5, type=int,
                        help="Number of training steps before motion starts")

    parser.add_argument("--train_steps", default=5, type=int,
                        help="Number of training steps per keyframe")

    parser.add_argument("--eval_skip", default=0, type=int,
                        help="Number of eval frames to skip")

    parser.add_argument("--exp_name", required=True,
                        type=str, help="Name of experiment")

    parser.add_argument("--depth_supervision", default=0.0,
                        type=float, help="Amount of depth supervision")

    parser.add_argument("--near_distance", default=0.3,
                        type=float, help="Distance from camera to ignore")

    parser.add_argument("--vis_frame", default=0,
                        type=int, help="visualize a specific frame")

    parser.add_argument("--save_path", type=str,
                        required=True,
                        help="Path to save directory where results are put")

    return parser.parse_args()

def rotation_z(degrees: float) -> np.ndarray:
    """Returns a rotation matrix that rotates around the z axis by degrees."""
    X = np.eye(4)
    X[:3, :3] = R.from_euler("z", degrees, degrees=True).as_matrix()
    return X

def translation_x(x: float) -> np.ndarray:
    """Returns a rotation matrix that rotates around the z axis by degrees."""
    X = np.eye(4)
    X[:3, 3] = np.array([x, 0, 0])
    return X

__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()

def set_frame(testbed, frame_idx: int, frame: Frame):
    depth_img = None
    depth_scale = 1.0
    if frame.depth_path is not None:
        depth_img = frame.get_depth()
        depth_scale = frame.depth_scale()*testbed.nerf.training.dataset.scale
    rgb = frame.get_rgb()
    rgb[...,0:3] = srgb_to_linear(rgb[...,0:3])
    testbed.nerf.training.set_image(frame_idx=frame_idx, img=rgb, depth_img=depth_img, depth_scale=depth_scale)
    testbed.nerf.training.set_camera_extrinsics(
        frame_idx=frame_idx, camera_to_world=frame.X_WV)
    testbed.nerf.training.set_camera_intrinsics(
        frame_idx=frame_idx, fx=frame.fx(), fy=frame.fy(), cx=frame.cx(), cy=frame.cy())

def rotate_frames(testbed, frames: List[Frame], degrees_per_keyframe: float):
    rot_z = rotation_z(degrees_per_keyframe)
    for frame in frames:
        X_WV = np.eye(4)
        X_WV[:3, :4] = frame.X_WV
        frame.X_WV = (rot_z @ X_WV)[:3, :4]

def translate_frames(testbed, frames: List[Frame], x: float):
    X = translation_x(x)
    for frame in frames:
        X_WV = np.eye(4)
        X_WV[:3, :4] = frame.X_WV
        frame.X_WV = (X @ X_WV)[:3, :4]

def resync_frames(testbed: Testbed, frames: List[Frame]):
    for i, frame in enumerate(frames):
        testbed.testbed.nerf.training.set_camera_extrinsics(
            frame_idx=i, camera_to_world=frame.X_WV)
        # testbed.nerf.training.set_camera_intrinsics(
        # 	frame_idx=i, fx=frame.fx(), fy=frame.fy(), cx=frame.cx(), cy=frame.cy())

SAVED_FRAMES = {

}

pool = ThreadPoolExecutor(max_workers=1)

def evaluate(testbed: Testbed, keyframe: int, test_frames: List[Frame], eval_skip: int, vis_frame: Frame, save_dir: Path):
    df = pd.DataFrame()
    # Write a vis frame
    if vis_frame:
        rgb_render = testbed.render_rgb(
            vis_frame.width(), vis_frame.height(), vis_frame.X_WV, vis_frame.fov_x(), spp=1)
        vis_file_name = f"{keyframe:04d}_{vis_frame.name}.png"
        rgb_render = (rgb_render*255).astype(np.uint8)
        def write_image():
            Image.fromarray(rgb_render).save(save_dir.joinpath(f"eval/{vis_file_name}.png"))
        pool.submit(write_image)
    

    test_frame_idxs = list(range(0, len(test_frames), eval_skip))
    for j in test_frame_idxs:
        eval_f = test_frames[j]
        eval_file_name = eval_f.rgb_path.stem

        # Groundtruth RGB
        if eval_f.rgb_path in SAVED_FRAMES:
            rgb_gt = SAVED_FRAMES[eval_f.rgb_path]
        else:
            rgb_gt = eval_f.get_rgb()
            if rgb_gt.shape[-1] == 4:
                rgb_gt = rgb_gt[...,0:3]*rgb_gt[...,3:4]
            SAVED_FRAMES[eval_f.rgb_path] = rgb_gt

        # Predicted RGB
        rgb_render = testbed.render_rgb(
            eval_f.width(), eval_f.height(), eval_f.X_WV, eval_f.fov_x(), spp=1)[..., :3]
        
        mse = compute_error("MSE", rgb_render, rgb_gt)
        psnr = mse2psnr(mse)
        ssim = compute_error("SSIM", rgb_render, rgb_gt)
        lpips = rgb_lpips(rgb_gt, rgb_render, 'alex', "cuda") 

        # Predicted Depth
        depth_mse = 0
        depth_mae = 0
        depth_mse_p = 0
        depth_mae_p = 0
        iou = 0
        if eval_f.depth_path is not None:
            depth_gt = eval_f.get_depth()
            depth_render = testbed.render_depth(
                eval_f.width(), eval_f.height(), eval_f.X_WV, eval_f.fov_x(), spp=1)
            # depth_mse = compute_error("MSE", depth_render, depth_gt)
            # depth_mae = compute_error("MAE", depth_render, depth_gt)
            mask = depth_gt > 0
            render_mask = depth_render > 0
            iou = np.sum(np.logical_and(mask, render_mask)) / np.sum(np.logical_or(mask, render_mask))

            # Full
            depth_mse = compute_error("MSE", depth_render, depth_gt)
            depth_mae = compute_error("MAE", depth_render, depth_gt)
            # Positive
            depth_mse_p = compute_error("MSE", depth_render[mask], depth_gt[mask])
            depth_mae_p = compute_error("MAE", depth_render[mask], depth_gt[mask])


        # Add information to dataframe
        df = df.append(
            {
                "depth_mse": depth_mse,
                "depth_mae": depth_mae,
                "depth_mse_p": depth_mse_p,
                "depth_mae_p": depth_mae_p,
                "iou": iou,
                "rgb_mse": mse,
                "rgb_psnr": psnr,
                "rgb_ssim": ssim,
                "rgb_lpips": lpips,
                "camera_name": eval_file_name
            }, ignore_index = True
        )
    return df


def create_save_directory(exp_name, save_path):
    exp_dir = Path(save_path).joinpath(exp_name)
    if exp_dir.exists():
        print("Experiment directory already exists. Overwriting...")
        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True)
    exp_dir.joinpath("eval").mkdir(parents=True)
    return exp_dir

def train_cycle(testbed: Testbed, keyframe: int, train_steps: int, test_frames: Optional[List[Frame]], eval_skip: int, vis_frame_idx:int, save: bool, save_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame()
    t0 = time.time()
    testbed.testbed.shall_train = True
    for _ in range(train_steps):
        testbed.testbed.frame()
    elapsed = time.time() - t0
    if save:
        testbed.testbed.shall_train = False
        df = evaluate(testbed, keyframe, test_frames, eval_skip, vis_frame_idx, save_dir)
        testbed.testbed.shall_train = True
        df.insert(0, "keyframe", keyframe)
        df.insert(0, "time_to_train", elapsed)
        df.insert(0, "train_steps", train_steps)
    return df

if __name__ == "__main__":
    args = parse_args()
    print(f"Running experiment {args.exp_name} with config {args.config}")

    # Create testbed
    manifest = Manifest(Path(args.scene))
    test_frames = None
    if args.save:
        test_manifest = Manifest(Path(args.test_scene))
        test_frames = test_manifest.frames

    # Configure testbed
    config = args.config
    testbed = Testbed(gui=args.gui, config=config)
    testbed.set_depth_supervision(args.depth_supervision)
    testbed.testbed.nerf.training.linear_colors = False
    testbed.testbed.nerf.training.random_bg_color = True
    testbed.testbed.nerf.training.density_grid_decay = 0.0
    testbed.testbed.nerf.render_with_camera_distortion = True
    testbed.testbed.nerf.cone_angle_constant = 0
    testbed.testbed.fov = 0
    testbed.testbed.optimize_particle_features = True
    testbed.testbed.optimize_particle_positions = True
    testbed.testbed.nerf.training.near_distance = args.near_distance
    testbed.testbed.use_physics = True
    testbed.testbed.nerf_scale = 4 
    testbed.testbed.velocity_damping = 0.96 
    testbed.testbed.nerf.render_min_transmittance = 1e-4
    testbed.testbed.background_color = [0.0, 0.0, 0.0, 1.0]

    df = pd.DataFrame()

    exp_dir = None
    if args.save:
        exp_dir = create_save_directory(args.exp_name, args.save_path)

    # Load scene
    testbed.load_scene(args.scene)
    testbed.testbed.set_camera_to_training_view(0)

    vis_frame = None
    if args.save:
        vis_frame = copy.deepcopy(test_frames[args.vis_frame])

    total_translation = args.max_translation/2 
    translate_by = float(args.translation_per_keyframe)
    # Train Dynamic Scene
    for j in tqdm(range(args.num_keyframes)):
        keyframe = j
            
        df2 = train_cycle(testbed, keyframe, args.train_steps if j>0 else args.static_train_steps, test_frames, args.eval_skip, vis_frame, args.save, exp_dir)
        if args.save:
            df = df.append(df2, ignore_index = True)

        rotate_frames(testbed, manifest.frames, args.degrees_per_keyframe)

        if total_translation > args.max_translation:
            translate_by *= -1 
            total_translation = 0
        translate_frames(testbed, manifest.frames, translate_by)
        total_translation += abs(translate_by)
        resync_frames(testbed, manifest.frames)
        if args.save:
            rotate_frames(testbed, test_frames, args.degrees_per_keyframe)
            translate_frames(testbed, test_frames, translate_by)

    # Write info
    if args.save:
        # save df
        print(f"Saving results to {exp_dir}")
        df.insert(0, "encoding", args.config)
        df.insert(0, "exp_name", args.exp_name)
        df.insert(0, "depth_scale", manifest.depth_scale)
        df.insert(0, "degrees_per_keyframe", args.degrees_per_keyframe)
        df.insert(0, "translation_per_keyframe", args.translation_per_keyframe)
        df.to_csv(exp_dir.joinpath("results.csv"), index=False)

    # Cleanup
    del testbed
