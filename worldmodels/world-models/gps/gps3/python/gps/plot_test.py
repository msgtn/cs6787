import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl
import sys
import glob
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
import gps



dir_exp = '/home/dev/scratch/gpscars/worldmodels/world-models/gps/gps3/experiments/car_world_badmm_experiment2/data_files/pol_sample_itr_*.pkl'
dir_exp = '/home/dev/scratch/gpscars/worldmodels/world-models/gps/gps3/experiments/car_world_badmm_experiment_parallel_10/data_files/pol_sample_itr_*.pkl'
policy_pkl_paths = glob.glob(dir_exp)
policy_pkl_paths.sort()
num_iterations = int(policy_pkl_paths[-1].split('_')[-1][:-4])+1
print(f"num_iterations:{num_iterations}")
all_rewards = [[0]]*num_iterations
for path in policy_pkl_paths:
    iteration_num = int(path.split('_')[-1][:-4])
    print(iteration_num)
    with open(path, 'rb') as infile:
        raw_data = pkl.load(infile)

    # raw_data is a list of gps.sample.sample_list.SampleList objects
    n_controllers = len(raw_data)
    rewards = [[-np.sum(sample._data['REWARD']) for sample in samplelist.get_samples()][0] for samplelist in raw_data]
    all_rewards[iteration_num] = rewards


fig, ax = plt.subplots()
ax.violinplot(all_rewards)
plt.show()
import pdb; pdb.set_trace()



