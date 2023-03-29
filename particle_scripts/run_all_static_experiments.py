import os 
import sys
import shutil

def do_system(arg):
	print(f"==== running: {arg}")
	err = os.system(arg)
	if err:
		print("FATAL: command failed")
		sys.exit(err)

if __name__ == "__main__":
    base_dir = "/media/jad/DATA/particle_nerf_experiments/synthetic_3"
    synthetic_dir = "/media/jad/DATA/datasets/nerf_synthetic"
    static_train_steps = 300
    experiments = [
        "chair",
        "drums",
        "ficus",
        "hotdog",
        "lego",
        "mic",
        "ship",
        "materials"
    ]

    num_keyframes = 100
    train_steps = 5
    eval_skip = 30

    # rotation experiments
    for exp in experiments: 
        for rot_speed in [1.0, 2.0, 3.0, 4.0]:
            for encoding in ["particle", "ngp"]:
                exp_name = f"{exp}_{encoding}_rot_{rot_speed}"
                do_system(f"python particle_scripts/run_static_testbed.py\
                    --num_keyframes {num_keyframes}\
                    --train_steps {train_steps}\
                    --static_train_steps {static_train_steps}\
                    --exp_name {exp_name}\
                    --scene {synthetic_dir}/{exp}/transforms_train.json\
                    --test_scene {synthetic_dir}/{exp}/transforms_test.json\
                    --save_path {base_dir}\
                    --translation_per_keyframe 0\
                    --max_translation 0\
                    --degrees_per_keyframe {rot_speed}\
                    --near_distance 0.3\
                    --vis_frame 35\
                    --eval_skip {eval_skip}\
                    --save\
                    --config {encoding}")
                do_system(f"ffmpeg -y -framerate 24 -pattern_type glob -i '{base_dir}/{exp_name}/eval/*.png' -c:v libx264 -pix_fmt yuv420p {base_dir}/videos/{exp_name}.mp4")

    # translation experiments
    for exp in experiments: 
        for trans_speed in [0.01, 0.02, 0.03]:
            for encoding in ["particle", "ngp"]:
                exp_name = f"{exp}_{encoding}_trans_{trans_speed}"
                do_system(f"python particle_scripts/run_static_testbed.py\
                    --num_keyframes {num_keyframes}\
                    --train_steps {train_steps}\
                    --static_train_steps {static_train_steps}\
                    --exp_name {exp_name}\
                    --scene {synthetic_dir}/{exp}/transforms_train.json\
                    --test_scene {synthetic_dir}/{exp}/transforms_test.json\
                    --save_path {base_dir}\
                    --translation_per_keyframe {trans_speed}\
                    --max_translation 1.0\
                    --degrees_per_keyframe {0}\
                    --near_distance 0.3\
                    --vis_frame 35\
                    --eval_skip {eval_skip}\
                    --save\
                    --config {encoding}")
                do_system(f"ffmpeg -y -framerate 24 -pattern_type glob -i '{base_dir}/{exp_name}/eval/*.png' -c:v libx264 -pix_fmt yuv420p {base_dir}/videos/{exp_name}.mp4")
