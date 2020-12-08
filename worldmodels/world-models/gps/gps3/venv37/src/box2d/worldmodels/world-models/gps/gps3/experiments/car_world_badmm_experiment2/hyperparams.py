# To get started, copy over hyperparams from another experiment.
# Visit rll.berkeley.edu/gps/hyperparams.html for documentation.

""" Hyperparameters for Box2d Car Racing BADMM."""
from __future__ import division

import os.path
from datetime import datetime
import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.box2d.agent_box2d import AgentBox2D
from gps.agent.box2d.car_world import CarWorld
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
# from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_car_world import CostCarWorld
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.gui.config import generate_experiment_info
from gps.proto.gps_pb2 import RGB_IMAGE, RGB_IMAGE_SIZE, ACTION, IMAGE_FEAT
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example import tf_network
from gps.algorithm.algorithm_badmm import AlgorithmBADMM

IMAGE_WIDTH = 8
IMAGE_HEIGHT = 4
IMAGE_CHANNELS = 1
NUM_FP = 15

SENSOR_DIMS = {
    ACTION: 3,
    "REWARD": 1,
    RGB_IMAGE: IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS,
    RGB_IMAGE_SIZE: 3,
    IMAGE_FEAT: NUM_FP * 2,  # affected by num_filters set below.
}
BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/car_world_badmm_experiment2/'


common = {
    'experiment_name': 'car_world_badmm_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 10,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentBox2D,
    'world' : CarWorld,
    'target_state' : None,
    'render' : True,
    'x0': np.zeros(IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS),
    # 'rk': 0,
    'dt': 0.01,
    'substeps': 1,
    'conditions': common['conditions'],
    'T': 500,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [RGB_IMAGE],
    'obs_include': [RGB_IMAGE],
}

algorithm = {
    'type': AlgorithmBADMM,
    'conditions': common['conditions'],
    'iterations': 10,
    'lg_step_schedule': np.array([1e-4, 1e-3, 1e-2, 1e-2]),
    'policy_dual_rate': 0.2,
    'ent_reg_schedule': np.array([1e-3, 1e-3, 1e-2, 1e-1]),
    'fixed_lg_step': 3,
    'kl_step': 5.0,
    'min_step_mult': 0.01,
    'max_step_mult': 1.0,
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1,
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains': np.zeros(SENSOR_DIMS[ACTION]),
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 0.1,
    'stiffness': 0.01,
    'dt': agent['dt'],
    'T': agent['T'],
}

cost_car_world = {
    'type': CostCarWorld,
    'wu': np.array([1, 1])
}


algorithm['cost'] = {
    'type': CostSum,
    'costs': [cost_car_world],
    'weights': [1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'weights_file_prefix': EXP_DIR + 'policy',
    'network_params': {
        'obs_include': [RGB_IMAGE],
        'sensor_dims': SENSOR_DIMS,
    },
    'network_model': tf_network
}



    # algorithm['policy_opt'] = {
    #     'type': PolicyOptTf,
    #     'network_params': {
    #         'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    #         'sensor_dims': SENSOR_DIMS,
    #     },
    #     'weights_file_prefix': EXP_DIR + 'policy',
    #     'iterations': 3000,
    #     'network_model': tf_network
    # }

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 20,
}

config = {
    'iterations': 20,
    'num_samples': 5,
    'verbose_trials': 5,
    'verbose_policy_trials': 5,
    'common': common,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)
