import subprocess
import os
os.environ["DISPLAY"]=":99"

max_controllers = 10
for i in range(1,max_controllers+1):
    experiment = f"car_world_badmm_controllers_{i}"
    subprocess.run(["xvfb-run", "-a","-s", "\"-screen 0 1400x900x24\"","python","/home/dev/scratch/gpscars/worldmodels/world-models/gps/gps3/python/gps/gps_main.py", experiment, "-c", f"{i}"])
