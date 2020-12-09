import subprocess

n_iters = 20
experiment = "car_world_badmm_experiment_sequential"
for i in range(n_iters):
    subprocess.run(["python","/home/dev/scratch/gpscars/worldmodels/world-models/gps/gps3/python/gps/gps_main.py", experiment, "-p", "1", f"{i}"])
