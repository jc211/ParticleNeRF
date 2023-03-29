import os 
import sys
from pathlib import Path
import json
import argparse

def do_system(arg):
	print(f"==== running: {arg}")
	err = os.system(arg)
	if err:
		print("FATAL: command failed")
		sys.exit(err)

PATH = Path(__file__).parent.absolute()

def generate_lyra_bash(job_name, command):
	# write bash script 
	script = f"""#!/bin/bash -l

	#PBS -N {job_name}
	#PBS -l ncpus=1
	#PBS -l ngpus=1
	#PBS -l gputype=A100
	#PBS -l mem=16gb
	#PBS -l walltime=20:00:00
	#PBS -j oe

	cd {PATH} 
	{command}
	"""

	# write script
	with open(f"{job_name}.sh", "w") as f:
		f.write(script)
	os.chmod(f"{job_name}.sh", 0o777)

	return f"qsub {job_name}.sh"

def generate_bash(job_name, path, command, conda_env):
	# write bash script 
	script = f"""#!/bin/bash
	cd {path} 
	conda activate {conda_env}	
	{command}
	"""

	# write script
	with open(f"{job_name}.sh", "w") as f:
		f.write(script)
	os.chmod(f"{job_name}.sh", 0o777)
	
	return f"source {job_name}.sh"


def parse_args():
	parser = argparse.ArgumentParser(description="Run neural graphics primitives testbed with additional configuration & output options")

	parser.add_argument("--computer", type=str, required=True,
						help="Computer configuration to use")
	parser.add_argument("--test", type=str, required=True, help="Test name to use")
	parser.add_argument("--encoding", type=str, required=True, help="Encoding to use")
	parser.add_argument("--save", action="store_true", help="Save the dataset")
	parser.add_argument("--save_vis", action="store_true", help="Render the visualization")
	parser.add_argument("--gui", action="store_true", help="Show the gui")
	return parser.parse_args()

def run_test(computer: str, test_name: str, enc:str, save: bool = True, save_vis: bool = True):
	# Get path to current file
	current_dir = Path(__file__).parent.absolute()

	# read experiments.json file 
	with open(f"{current_dir}/experiments.json", "r") as f:
		experiments = json.load(f)
	
	# check if computer exists in config
	if computer not in experiments["computers"]:
		print(f"Computer {computer} not found in experiments.json")
		sys.exit(1)
	
	# check if testname exists
	if test_name not in experiments["tests"]:
		print(f"Test {test_name} not found in experiments.json")
		sys.exit(1)
	
	save_path = experiments["computers"][computer]["save_path"]
	base_path = experiments["computers"][computer]["base_path"]

	model_name = experiments["tests"][test_name]["model"]

	# check if test exists in data_path
	if not Path(f"{base_path}/{model_name}").exists():
		print(f"Test {test_name} - {model_name} not found in {base_path}")
		sys.exit(1)
	
	# check if encoding requested is supported
	encodings = ["ngp", "particle", "tnv", "tnv-s"]
	if enc not in encodings:
		print(f"Encoding {enc} not found in {encodings}")
		sys.exit(1)


	base_path = base_path + "/" + model_name
	save_path = save_path + "/"  + test_name

	# combine args from dictionary
	exp_args = {**experiments["global"], **experiments["global"]["encodings"][enc], **experiments["tests"][test_name]}

	# pop encodings args
	encodings_args = exp_args.pop("encodings")

	# add encoding args if it matches with requested encoding
	if enc in encodings_args:
		exp_args = {**exp_args, **encodings_args[enc]}

	# flatten to kwargs string
	all_args = [f"--{k} {v}" if type(v) != bool else f"--{k}" for k, v in exp_args.items()]
	# combine into string
	all_args = " ".join(all_args)


	save_args = ""
	if save:
		save_args += " --save"
	if save_vis:
		save_args += " --save_vis"
	
	if args.gui:
		save_args += " --gui"
	
	all_args += f" {save_args}" 
	exp_name = f"{test_name}-{enc}"

	if enc in ["particle", "ngp"]:
		py_file = "run_testbed.py" 
	else:
		py_file = "TiNeuVox/run_testbed.py" 

	conda_env = "instant-ngp" if enc in ["particle", "ngp"] else "tineuvox"
	description = f"{computer}_{test_name}_{enc}"

	system_command = f"conda run --live-stream -n {conda_env} python {py_file}\
		--scene {base_path}/transforms.json\
		--save_path {save_path}\
		--exp_name {exp_name}\
		--description '{description}'\
		{all_args}"
	
	if args.save_vis:
		video_command_1 = f"ffmpeg -y -framerate 24 -pattern_type glob -i '{save_path}/{exp_name}/eval/*.png' -c:v libx264 -pix_fmt yuv420p {save_path}/{exp_name}/{exp_name}.mp4"
		video_command_2 = f"ffmpeg -y -framerate 24 -pattern_type glob -i '{save_path}/{exp_name}/eval_path/*.png' -c:v libx264 -pix_fmt yuv420p {save_path}/{exp_name}/{exp_name}_path.mp4"

		video_command_3 = f"ffmpeg -y -framerate 24 -pattern_type glob -i '{save_path}/{exp_name}/eval_depth/*.png' -c:v libx264 -pix_fmt yuv420p {save_path}/{exp_name}/{exp_name}_depth.mp4"
		video_command_4 = f"ffmpeg -y -framerate 24 -pattern_type glob -i '{save_path}/{exp_name}/eval_path_depth/*.png' -c:v libx264 -pix_fmt yuv420p {save_path}/{exp_name}/{exp_name}_path_depth.mp4"
		command = f"{system_command} && {video_command_1} && {video_command_2} && {video_command_3} && {video_command_4}"
	else:
		command = system_command

	if computer == "lyra":
		command = generate_lyra_bash(exp_name, command)
	
	do_system(command)

if __name__ == "__main__":
	args = parse_args()
	run_test(args.computer, args.test, args.encoding, args.save, args.save_vis)