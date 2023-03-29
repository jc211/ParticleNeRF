import os 
import sys

def do_system(arg):
	print(f"==== running: {arg}")
	err = os.system(arg)
	if err:
		print("FATAL: command failed")
		sys.exit(err)

if __name__ == "__main__":
    computer_name = "main"

    tests = [
        "robot-task",
        "robot",
        "cloth",
        "spring",
        "pendulums",
        "wheel",

        "wheel_speed",
        "wheel_no_positions", 
        "wheel_no_features",

        "wheel_speed_t10",
        "wheel_speed_t15",
        "wheel_speed_t20",
        "wheel_speed_t40",

        "wheel_part_50k",
        "wheel_part_100k",
        "wheel_part_200k",
        "wheel_part_300k",

        "wheel_search_3",
        "wheel_search_4",
        "wheel_search_5",
        "wheel_search_6",

        "wheel_grad_05", 
        "wheel_grad_10", 
        "wheel_grad_20", 
        "wheel_grad_30", 
        "wheel_grad_40", 
        "wheel_grad_50", 
        "wheel_grad_60", 
        "wheel_grad_80", 

        "wheel_adam_005",
        "wheel_adam_01",
        "wheel_adam_02",
        "wheel_adam_05",
        "wheel_adam_1",
        "wheel_adam_2",
        "wheel_adam_3",
        "wheel_adam_4",

        "wheel_long_running",
        ]
    encs = ["ngp", "particle"]
    for test in tests:
        for enc in encs:
            do_system(f"python run_experiment.py --computer {computer_name} --test {test} --encoding {enc} --save_vis --save")
    
