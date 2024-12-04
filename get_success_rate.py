import glob, os, sys, json


exp_dir = sys.argv[1]
exp_files = glob.glob(exp_dir + "/run_*/steps*.json")

n_steps_per_run, success_rate = 0.0, 0.0

for exp_file in exp_files:
    data = json.load(open(exp_file, "r"))
    if data["success"]:
        success_rate += 1
        if "step" in data.keys():
            n_steps_per_run += data["step"]
        if "steps" in data.keys():
            n_steps_per_run += data["steps"]
print(f"Average number of steps per success: {n_steps_per_run/success_rate}")
print(f"Total number of runs: {len(exp_files)}")
print(f"Success Rate: {success_rate/len(exp_files)}")




