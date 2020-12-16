import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import pickle as pkl
import sys
import glob
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
import gps
import json

labels = []
def add_label(violin, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))

# conditions = ["adagrad", "adam_no_batch_norm_09", "rmsprop_no_batch_norm", "sgd_no_batch_norm"]
# legend_labels = ["Adagrad", "Adam", "RMSprop", "SGD"]
# cmap = ['red', 'yellow', 'green', 'blue' ]
conditions = ["adam_batch_norm", "adam_batch_norm_except_layer1", "adam_no_batch_norm_09"]
legend_labels = ["Batch normalization at each layer", "Batch normalization at each layer except first layer", "No batch normalization"]

fig, ax = plt.subplots()
ax.set_xlabel("Training Iteration Number")
ax.set_ylabel("Distribution of Test Rewards After Each Iteration of GPS")
ax.set_xticks(np.arange(0,21,1))
ax.set_ylim([-50, 60])
for i in range(0,len(conditions)):
    dir_exp = '/media/alap/OS/Users/alap_/CatkinWorkspaces/cs6787/worldmodels/world-models/gps/gps3/experiments/car_world_badmm_experiment2/data_files/new_studies/'+ conditions[i] + '/pol_sample_itr_*.pkl'
# dir_exp = '/media/alap/OS/Users/alap_/CatkinWorkspaces/cs6787/worldmodels/world-models/gps/gps3/experiments/car_world_badmm_experiment2/data_files/new_studies/rmsprop_no_batch_norm/pol_sample_itr_*'

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
    add_label(plt.violinplot(all_rewards), legend_labels[i])
    ##Average reward
    total_reward = [item for sublist in all_rewards for item in sublist]
    print("Average reward:", sum(total_reward)/len(total_reward))
    with open('/media/alap/OS/Users/alap_/CatkinWorkspaces/cs6787/worldmodels/world-models/gps/gps3/experiments/car_world_badmm_experiment2/data_files/new_studies/'+conditions[i]+'/timestamps.json') as f:
        timestamps = json.load(f)["local_controller_times"]
        print("Average time per iteration:", sum(timestamps)/len(timestamps))
ax.legend(*zip(*labels), loc=2)
plt.show()
    # import pdb; pdb.set_trace()
