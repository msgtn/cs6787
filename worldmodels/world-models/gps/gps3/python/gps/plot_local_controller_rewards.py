import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl
import sys
import glob
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
import gps
from scipy import stats

'''
H3b: plot the test rewards for each policy trained using different number of local controllers
The rewards are stored in policy.pkl files
'''

last_iter_rewards = []
max_local_controllers = 6

for i in range(1,max_local_controllers+1):
    print("num_workers: ",i)
    dir_exp = f'/home/dev/scratch/gpscars/worldmodels/world-models/gps/gps3/experiments/car_world_badmm_controllers_{i}/data_files/pol_sample_itr_*.pkl'
    policy_pkl_paths = glob.glob(dir_exp)
    policy_pkl_paths.sort()
    num_iterations = int(policy_pkl_paths[-1].split('_')[-1][:-4])+1
    print(f"num_iterations:{num_iterations}")
    all_rewards = [[0]]*num_iterations
    for path in policy_pkl_paths:
        iteration_num = int(path.split('_')[-1][:-4])
        print(iteration_num)
        with open(path, 'rb') as infile:
            raw_data = pkl.load(infile) # raw_data is a list of gps.sample.sample_list.SampleList objects
        n_controllers = len(raw_data)
        rewards = [[-np.sum(sample._data['REWARD']) for sample in samplelist.get_samples()][0] for samplelist in raw_data]
        all_rewards[iteration_num] = rewards
    last_iter_rewards.append(all_rewards[-1])

flattened_rewards = [(i,reward) for experiment in all_rewards for i, reward in enumerate(experiment)]

x,y = zip(*flattened_rewards)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(f"y = {slope}x + {intercept}, r^2={r_value**2}, p={p_value}")
fig, ax = plt.subplots()
ax.violinplot(last_iter_rewards)
plt.xticks(range(1,max_local_controllers+1))
plt.xlabel("Number of Local Controllers/Workers")
plt.ylabel("Distribution of Test Rewards After Last Iteration of GPS")
plt.show()




