from pathlib import Path
import argparse
from typing import List
import numpy as np
from testbed import Testbed
from manifest import Frame, Manifest
import time
import shutil
import json
from PIL import Image
import pandas as pd
from common import *
import torch
from concurrent.futures import ThreadPoolExecutor

CONFIG_PATH = Path(__file__).parents[1].joinpath("configs/nerf")

def parse_args():
    parser = argparse.ArgumentParser(description="Run neural graphics primitives testbed with additional configuration & output options")

    parser.add_argument("--scene", "--training_data", required=True,
                        help="The scene to load. Can be the scene's name or a full path to the training data.")

    parser.add_argument("--test_scene", default="", help="The scene to test on")

    parser.add_argument("--config", default="base",
                        help="Name of config file to use")

    parser.add_argument("--n_particles", type=int, nargs='+', help="Number of particles to use")
    parser.add_argument("--search_radius", type=float, nargs='+', help="Search radii to use")
    parser.add_argument("--n_levels", type=int, help="Number of levels to use")
    parser.add_argument("--n_cameras", type=int, help="Number of cameras used in each animation frame")
    parser.add_argument("--position_learning_rate", type=float)
    parser.add_argument("--collision_min_distance", default=0.01, type=float)
    parser.add_argument("--model", default="unspecified", type=str)

    parser.add_argument("--speeds", type=int, nargs='+')
    parser.add_argument("--speeds_duration", type=int, nargs='+')

    parser.add_argument("--gui", action="store_true",
                        help="Run the testbed GUI interactively.")

    parser.add_argument("--no_feat", action="store_true")
    parser.add_argument("--no_pos", action="store_true")
    parser.add_argument("--no_physics", action="store_true")
    parser.add_argument("--no_collision", action="store_true")

    parser.add_argument("--save", action="store_true", help="Save the dataset")
    parser.add_argument("--save_vis", action="store_true", help="Render the visualization")
    parser.add_argument("--just_time", action="store_true", help="Render the visualization")
    parser.add_argument("--vis_every", default=1, type=int, help="Render the visualization every n steps")
    parser.add_argument("--render_radius", default=3, type=float, help="How far to render the flyover")
    parser.add_argument("--max_eval", default=0, type=int, help="How many frames to use for evaluation")

    parser.add_argument("--train_steps", default=5, type=int,
                        help="Number of training steps per keyframe")

    parser.add_argument("--eval_skip", default=0, type=int,
                        help="Number of eval frames to skip")

    parser.add_argument("--exp_name", required=True,
                        type=str, help="Name of experiment")

    parser.add_argument("--depth_supervision", default=0.0,
                        type=float, help="Amount of depth supervision")

    parser.add_argument("--near_distance", "--near", default=0.3,
                        type=float, help="Distance from camera to ignore")
        
    parser.add_argument("--far", default=0.3,
                        type=float, help="Does nothing")

    parser.add_argument("--vis_frame", default=0,
                        type=int, help="visualize a specific frame")

    parser.add_argument("--save_path", type=str,
                        required=True,
                        help="Path to save directory where results are put")

    parser.add_argument("--description", type=str, default="",
                        help="Description of experiment")
    return parser.parse_args()


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

def srgb_to_linear(img):
    limit = 0.04045
    return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)

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

pool = ThreadPoolExecutor(max_workers=1)

def visualize_frame(testbed, animation_step, frame: Frame, save_dir, X_WV = None): 
    folder = "eval" if X_WV is None else "eval_path"
    X_WV = frame.X_WV if X_WV is None else X_WV
    rgb_render = testbed.render_rgb(
        frame.width(), frame.height(), X_WV, frame.fov_x(), spp=1)
    vis_file_name = f"{animation_step:04d}_{frame.name}.png"
    rgb_render = (rgb_render*255).astype(np.uint8)

    depth_render = testbed.render_depth(frame.width(), frame.height(), frame.X_WV, frame.fov_x(), spp=1)
    def write_image():
        Image.fromarray(rgb_render).save(save_dir.joinpath(f"{folder}/{vis_file_name}"))

    depth_render = ((depth_render / 10.0)*65535).astype(np.uint16)
    def write_depth():
        Image.fromarray(depth_render).save(save_dir.joinpath(f"{folder}_depth/{vis_file_name}"))
    pool.submit(write_depth)
    pool.submit(write_image)

def evaluate(testbed: Testbed, test_frames: List[Frame]) -> pd.DataFrame:
    dfs = []
    for j in range(len(test_frames)):
        eval_f = test_frames[j]
        eval_file_name = eval_f.rgb_path.stem

        rgb_gt = eval_f.get_rgb()
        if rgb_gt.shape[-1] == 4:
            rgb_gt = rgb_gt[...,0:3]*rgb_gt[...,3:4]

        # Predicted RGB
        rgb_render = testbed.render_rgb(
            eval_f.width(), eval_f.height(), eval_f.X_WV, eval_f.fov_x(), spp=1)[..., :3]
        rgb_render = np.clip(rgb_render, 0, 1)

        # Convert to linear
        mse = compute_error("MSE", rgb_render, rgb_gt)
        psnr = mse2psnr(mse)
        ssim = compute_error("SSIM", rgb_render, rgb_gt)
        lpips = rgb_lpips(rgb_gt, rgb_render, 'alex', "cuda") 

        # Predicted Depth
        depth_mse = 0
        depth_mae = 0
        if eval_f.depth_path is not None:
            depth_scale = eval_f.depth_scale()
            depth_gt = eval_f.get_depth()*depth_scale
            depth_render = testbed.render_depth(
                eval_f.width(), eval_f.height(), eval_f.X_WV, eval_f.fov_x(), spp=1)

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
        dfs.append(
            pd.DataFrame({
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
            }, index=[0])
        )
    return pd.concat(dfs, ignore_index=True)

def create_save_directory(exp_name, save_path):
    exp_dir = Path(save_path).joinpath(exp_name)
    if exp_dir.exists():
        print("Experiment directory already exists. Overwriting...")
        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True)
    exp_dir.joinpath("eval").mkdir(parents=True)
    exp_dir.joinpath("eval_path").mkdir(parents=True)
    exp_dir.joinpath("eval_depth").mkdir(parents=True)
    exp_dir.joinpath("eval_path_depth").mkdir(parents=True)
    return exp_dir

def frames_for_step(train_frames, step, n_frames_per_step):
    if step > (len(train_frames)/n_frames_per_step - 1):
        step = int(step % (len(train_frames)/n_frames_per_step))
    start = step*n_frames_per_step
    end = (step+1)*n_frames_per_step
    return train_frames[start:end]

def step_animation(testbed, frames): 
    for i, f in enumerate(frames):
        set_frame(testbed.testbed, i, f) 

def get_frame(frames: List[Frame], frame_idx: int, n_cameras: int, camera_num:int) -> Frame:
    id = frame_idx*n_cameras + camera_num
    return frames[id]

def train_cycle(testbed: Testbed, train_steps: int):
    timings = {
        "grid": 0,
        "forward": 0,
        "backward": 0,
        "physics": 0,
        "training_total": 0,
        "total_python": 0,
    }
    testbed.testbed.shall_train = True
    t0 = time.time()
    for _ in range(train_steps):
        testbed.testbed.frame()
        timings["grid"] += testbed.testbed.grid_time_ms()
        timings["forward"] += testbed.testbed.forward_time_ms()
        timings["backward"] += testbed.testbed.backward_time_ms()
        timings["physics"] += testbed.testbed.physics_time_ms()
        timings["training_total"] += testbed.testbed.training_time_ms()
    elapsed = time.time() - t0
    timings["total_python"] = elapsed*1000
    testbed.testbed.shall_train = False
    return timings 

def look_at_transform(eye, target, up=np.array([0.0, 0.0, 1.0])):
    eye = np.asarray(eye)
    target = np.asarray(target)
    up = np.asarray(up)
    up /= np.linalg.norm(up)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    X_WV = np.eye(4)
    X_WV[:3, 0] = x
    X_WV[:3, 1] = y
    X_WV[:3, 2] = z
    X_WV[:3, 3] = eye
    return X_WV


def pose_generator(t, r):
    period =  280 
    return look_at_transform(eye=[r*np.cos(-2*np.pi*t/period), r*np.sin(-2*np.pi*t/period), 2.1], target=[0.0, 0.0, 0.0])

if __name__ == "__main__":
    args = parse_args()
    print(f"Running experiemt {args.exp_name} with config {args.config}")


    if not Path(args.scene).exists():
        raise ValueError(f"Scene {args.scene} does not exist")
    # load json file
    with open(args.scene, "r") as f:
        scene = json.load(f)
        scene["n_frames"] = args.eval_skip
    # write json file
    with open(args.scene, "w") as f:
        json.dump(scene, f)

    manifest = Manifest(Path(args.scene))

    # Eval Poses
    eval_poses_path = Path(args.scene).parent.joinpath("eval_path.json")
    eval_poses = None
    if eval_poses_path.exists():
        print("Found evaluation poses")
        with open(eval_poses_path, "r") as f:
            eval_poses = json.load(f)

    n_frames_per_animation_step = args.n_cameras
    if args.eval_skip > n_frames_per_animation_step or args.eval_skip < 1: 
        raise ValueError(f"Eval skip must be less than or equal to the number of frames per animation step ({n_frames_per_animation_step})")
    n_training_frames_per_animation_step = args.eval_skip 
    n_test_frames_per_animation_step = n_frames_per_animation_step - n_training_frames_per_animation_step 
    print(f"Training on {n_training_frames_per_animation_step} frames per animation step, evaluating on {n_test_frames_per_animation_step} frames per animation step")
    test_frames = []
    train_frames = []

    for i in range(len(manifest.frames)):
        train_frames.extend(manifest.frames[i*n_frames_per_animation_step:i*n_frames_per_animation_step + args.eval_skip])
        test_frames.extend(manifest.frames[args.eval_skip + i*n_frames_per_animation_step: (i+1)*n_frames_per_animation_step])
    print(f"Training on {len(train_frames)} frames, evaluating on {len(test_frames)} frames")


    # Configure testbed
    config = args.config
    with open(CONFIG_PATH.joinpath(f"{config}.json"), 'r') as f:
        local_config = json.load(f)

    if config == "particle":
        if args.n_particles is not None: 
            local_config["encoding"]["n_particles"] = args.n_particles
        if args.search_radius is not None: 
            local_config["encoding"]["search_radius"] = args.search_radius
        if args.n_levels is not None: 
            local_config["encoding"]["n_levels"] = args.n_levels
        if args.position_learning_rate is not None:
            local_config["encoding"]["position_optimizer"]["learning_rate"] = args.position_learning_rate

    # Create testbed
    testbed = Testbed(gui=args.gui, config=local_config)
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
    testbed.testbed.nerf.render_min_transmittance = 1e-4
    testbed.testbed.background_color = [0.0, 0.0, 0.0, 1.0]
    testbed.testbed.nerf_scale = 2
    testbed.testbed.use_physics = True
    if args.position_learning_rate is not None:
        testbed.testbed.nerf_scale = args.position_learning_rate
    testbed.testbed.velocity_damping = 0.96 
    if args.no_physics is not None and args.no_physics:
        print("Disabling physics")
        testbed.testbed.use_physics = False

    testbed.testbed.physics_min_distance = args.collision_min_distance
    testbed.set_n_training_images(n_training_frames_per_animation_step)

    exp_dir = None
    if args.save or args.save_vis:
        exp_dir = create_save_directory(args.exp_name, args.save_path)

    if args.save:
        if config == "particle":
            exp_df = pd.DataFrame({
                "exp_name": [args.exp_name],
                "config": [args.config],
                "model": [args.model],
                "n_levels": [local_config["encoding"]["n_levels"]],
                "speeds": [np.array(args.speeds, dtype=np.int32)],
                "speeds_duration": [np.array(args.speeds_duration, dtype=np.int32)],
                "search_radius": [np.array(local_config["encoding"]["search_radius"], dtype=np.float32)],
                "n_particles": [np.array(local_config["encoding"]["n_particles"], dtype=np.int32)],
                "learning_rate": [local_config["encoding"]["position_optimizer"]["learning_rate"]],
            }, index=[0])
        else:
            exp_df = pd.DataFrame({
                "exp_name": [args.exp_name],
                "model": [args.model],
                "config": [args.config],
                "speeds": [np.array(args.speeds, dtype=np.int32)],
                "speeds_duration": [np.array(args.speeds_duration, dtype=np.int32)],
            }, index=[0])

        exp_df.insert(0, "depth_scale", manifest.depth_scale)
        exp_df.to_csv(exp_dir.joinpath("exp_info.csv"))


    # Load scene
    testbed.load_scene(args.scene)
    testbed.testbed.set_camera_to_training_view(0)

    dfs = []
    animation_step = 0
    keyframe = 0
    step_test_frames = frames_for_step(test_frames, keyframe, n_test_frames_per_animation_step)
    for i, speed_duration in enumerate(args.speeds_duration):
        for j in range(speed_duration):
            print(f"{args.exp_name} - Animation step {animation_step}")
            speed = args.speeds[i]
            if speed != 0:
                step_train_frames = frames_for_step(train_frames, keyframe, n_training_frames_per_animation_step)
                step_animation(testbed, step_train_frames)
                if args.no_feat:
                    print("Disabling feature optimization")
                    testbed.testbed.optimize_particle_features = False
                if args.no_pos:
                    print("Disabling position optimization")
                    testbed.testbed.optimize_particle_positions = False

            timings = train_cycle(testbed, args.train_steps)
            if args.save:
                if len(step_test_frames) > 0:
                    if not args.just_time:
                        step_test_frames = frames_for_step(test_frames, keyframe, n_test_frames_per_animation_step)
                        if args.max_eval != 0:
                            step_test_frames = step_test_frames[:args.max_eval]
                        df = evaluate(testbed, step_test_frames)
                        df.insert(0, "keyframe", keyframe)
                        df.insert(0, "animation_step", animation_step)
                        df.insert(0, "time_to_train", timings["total_python"])
                        df.insert(0, "physics_time_ms", timings["physics"])
                        df.insert(0, "grid_time_ms", timings["grid"])
                        df.insert(0, "forward_time_ms", timings["forward"])
                        df.insert(0, "backward_time_ms", timings["backward"])
                        df.insert(0, "training_time_ms", timings["training_total"])
                        df.insert(0, "speed", speed)
                    else:
                        df = pd.DataFrame({
                            "keyframe": [keyframe],
                            "animation_step": [animation_step],
                            "time_to_train": [timings["total_python"]],
                            "physics_time_ms": [timings["physics"]],
                            "grid_time_ms": [timings["grid"]],
                            "forward_time_ms": [timings["forward"]],
                            "backward_time_ms": [timings["backward"]],
                            "training_time_ms": [timings["training_total"]],
                            "speed": [speed],
                        }, index=[0])
                    dfs.append(df)
                else:
                    print(f"No test frames for this animation step - keyframe {keyframe}")

            if not args.just_time and args.save_vis and speed != 0 and animation_step % args.vis_every == 0:
                visualize_frame(testbed, animation_step, test_frames[args.vis_frame], exp_dir, pose_generator(animation_step, args.render_radius)[:3, :])
                visualize_frame(testbed, animation_step, test_frames[args.vis_frame - args.eval_skip], exp_dir)

            animation_step += 1
            keyframe += speed

    if args.save:
        df = pd.concat(dfs)
        df.insert(0, "exp_name", args.exp_name)
        df.to_csv(exp_dir.joinpath("results.csv"), index=False)