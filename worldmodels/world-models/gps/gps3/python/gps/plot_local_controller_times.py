import numpy as np
import json
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

'''
H3a: plot the local controller and total iteration runtimes for different numbers of local controllers
The rewards are stored in policy.pkl files
'''

base = '/home/dev/scratch/gpscars/worldmodels/world-models/gps/gps3/experiments/car_world_badmm_controllers_'
timestamps = dict()
local_times = []
local_times_chunked = []
iteration_times = []
iteration_times_chunked = []
total_times = []
for i in range(11):
    try:
        with open(base + str(i) + '/data_files/timestamps.json') as infile:
            timestamps[i] = ts = json.load(infile)
            total_times.append((i,ts['end']-ts['start']))
            local_times_chunked.append(ts['local_controller_times'])
            iteration_times_chunked.append(ts['iteration_times'])
            for j, lt in enumerate(ts['local_controller_times']):
                local_times.append((i,lt))
                iteration_times.append((i,ts['iteration_times'][j]))
    except FileNotFoundError:
        print(f"No file for {i}")

# run a quick linear regression for the local controller rollouts vs num_workers
x,y = zip(*local_times)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(f"y = {slope}x + {intercept}, r^2={r_value**2}, p={p_value}")
print(f"Average time for one controller:{np.mean(local_times_chunked[0])}")

labels = []
violin = plt.violinplot(local_times_chunked)
local_color = violin["bodies"][0].get_facecolor().flatten()
labels.append((mpatches.Patch(color=local_color), 'Local Controller Rollouts Only'))
violin = plt.violinplot(iteration_times_chunked)
iteration_color = violin["bodies"][0].get_facecolor().flatten()
labels.append((mpatches.Patch(color=iteration_color), 'Complete Iteration'))

plt.plot(range(1, len(local_times_chunked)+1),[np.mean(chunk) for chunk in local_times_chunked], marker = 'o', color=local_color)
plt.plot(range(1, len(iteration_times_chunked)+1),[np.mean(chunk) for chunk in iteration_times_chunked], marker = 'o', color=iteration_color)

plt.xlabel('Number of Local Controllers/Workers')
plt.xticks(range(1, len(local_times_chunked)+1))
plt.ylabel('Runtime Per Iteration (s)')
plt.legend(*zip(*labels),loc="upper left")
plt.show()
