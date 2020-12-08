""" This file defines the cost for the car world. """
import copy

import numpy as np

# from gps.algorithm.cost.config import COST_ACTION
from gps.algorithm.cost.cost import Cost


class CostCarWorld(Cost):
    """ Computes torque penalties. """
    def __init__(self, hyperparams):
        # config = copy.deepcopy()
        # config.update(hyperparams)
        Cost.__init__(self, None)

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample: A single sample
        """
        sample_u = sample.get_U()
        T = sample.T
        dU = sample.dU
        dX = sample.dX
        l = np.zeros(T)
        lu = np.zeros((T, dU))
        lx = np.zeros((T, dX))
        luu = np.zeros((T, dU, dU))
        lxx = np.zeros((T, dX, dX))
        lux = np.zeros((T, dU, dX))
        l = sample.get("REWARD").reshape(T)
        return l, lx, lu, lxx, luu, lux
        
        l = 0.5 * np.sum(self._hyperparams['wu'] * (sample_u ** 2), axis=1)
        lu = self._hyperparams['wu'] * sample_u
        lx = np.zeros((T, Dx))
        luu = np.tile(np.diag(self._hyperparams['wu']), [T, 1, 1])
        lxx = np.zeros((T, Dx, Dx))
        lux = np.zeros((T, Du, Dx))

        # return np.zeros(l.shape()), np.zeros(lx.shape()), np.zeros(lu.shape()), np.zeros(lxx.shape()), luu, lux
        return l, lx, lu, lxx, luu, lux
