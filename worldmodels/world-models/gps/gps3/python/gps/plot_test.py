import numpy as np
import matplotlib
# matplotlib.use('agg')
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import pickle as pkl
import sys
import glob
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
import gps



dir_exp = '/home/dev/scratch/gpscars/worldmodels/world-models/gps/gps3/experiments/car_world_badmm_experiment2/data_files/pol_sample_itr_*.pkl'
dir_exps = [
    '/Users/PsychoMugs/projects/gpscar/worldmodels/world-models/gps/gps3/experiments/car_world_badmm_workers_3/pol_sample_itr_*.pkl',
    '/Users/PsychoMugs/projects/gpscar/worldmodels/world-models/gps/gps3/experiments/car_world_dream_experiment/data_files_02/pol_sample_itr_*.pkl',
    '/Users/PsychoMugs/projects/gpscar/worldmodels/world-models/gps/gps3/experiments/car_world_dream_experiment/data_files_/pol_sample_itr_*.pkl',
    ]
legends = ['Baseline', 'VAE (rescale)', 'VAE (crop-resample)']
plot_handles = []
fig, ax = plt.subplots()
labels = []
def add_label(violin, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))
max_len = 0

for dir_exp,l in zip(dir_exps,legends):
    policy_pkl_paths = glob.glob(dir_exp)
    policy_pkl_paths.sort()
    num_iterations = int(policy_pkl_paths[-1].split('_')[-1][:-4])+1
    print(f"num_iterations:{num_iterations}")
    all_rewards = [[0]]*num_iterations
    max_len = max(max_len,len(policy_pkl_paths))
    for path in policy_pkl_paths:
        iteration_num = int(path.split('_')[-1][:-4])
        print(iteration_num)
        with open(path, 'rb') as infile:
            raw_data = pkl.load(infile)

        # raw_data is a list of gps.sample.sample_list.SampleList objects
        n_controllers = len(raw_data)
        rewards = [[-np.sum(sample._data['REWARD']) for sample in samplelist.get_samples()][0] for samplelist in raw_data]
        all_rewards[iteration_num] = rewards


    # plot_handles.append(plt.violinplot(all_rewards))
    add_label(plt.violinplot(all_rewards), l)

plt.xlabel('Iteration')
plt.ylabel('Distribution of Test Rewards')
# plt.legend(*zip(plot_handles, legends))
plt.legend(*zip(*labels), loc=2)
plt.xticks(list(range(max_len+1)))
plt.savefig('/Users/PsychoMugs/projects/gpscar/worldmodels/world-models/gps/gps3/experiments/car_world_dream_experiment/data_files_02/vae_plot.jpg')
plt.show()
# import pdb; pdb.set_trace()

