import numpy as np
import json
from matplotlib import pyplot as plt
from scipy import stats
'''
H3c: plot the speedup vs theoretical speedup as we increase workers but keep num local controllers fixed
The rewards are stored in policy.pkl files
'''
base = '/home/dev/scratch/gpscars/worldmodels/world-models/gps/gps3/experiments/car_world_badmm_workers_'
timestamps = dict()
local_times = []
iteration_times = []
total_times = []
max_workers = 10

for i in range(max_workers+1):
    try:
        with open(base + str(i) + '/data_files/timestamps.json') as infile:
            timestamps[i] = ts = json.load(infile)
            total_times.append((i,ts['end']-ts['start']))
            for j, lt in enumerate(ts['local_controller_times']):
                local_times.append((i,lt))
                iteration_times.append((i,ts['iteration_times'][j]))
    except FileNotFoundError:
        print(f"No file for {i}")

# Amdahl's law: speedup = 1/((1-p) + p/s)
local_controller_baseline = timestamps[1]['local_controller_times']
total_iterations_baseline = timestamps[1]['iteration_times']
p = np.mean([t/total_iterations_baseline[i] for i,t in enumerate(local_controller_baseline)])
sd_p = np.std([t/total_iterations_baseline[i] for i,t in enumerate(local_controller_baseline)])
print(f'Proportion of time that is parallizable: {p}, std: {sd_p}')
theoretical_speedups = [1/((1-p) + (p/s)) for s in range(2,11)]


x,y = zip(*local_times)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
serial_t = total_times[0][1]
raw_speedups = [serial_t/par_t[1] for par_t in total_times[1:]]
speedups = [(serial_t/par_t[1])/(par_t[0]-1) for par_t in total_times[1:]]
n_workers = [t[0] for t in total_times[1:]]
plt.plot(n_workers, raw_speedups, marker='o', label='Observed Speedup')
plt.plot(n_workers, theoretical_speedups, marker='o', label='Theoretical Speedup (Amdahl\'s Law)')
plt.xlabel('Number of Workers')
plt.ylabel('Speedup over Sequential Baseline')
plt.legend(loc=0)
plt.show()
