"""ガウス過程のテンプレートの実装"""
import numpy as np
from abc import ABCMeta, abstractmethod

from ..kernel import RBF
from ..mean import Zero

class MyModel(metaclass=ABCMeta):

    def __init__(self, name, mean=None, kernel=None, input_dim=1, seed=None):
        self.name = name
        self.mean = mean if mean is not None else Zero()
        self.kernel = kernel if kernel is not None else RBF()
        self.input_dim = input_dim
        self.rand = np.random.RandomState(seed=seed)

        self.n_points = 0 
        self.n_params = self.mean.n_params + self.kernel.n_params
    
    @abstractmethod
    def add_points(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def sampling(self):
        raise NotImplementedError()

    def optimize(self):
        raise NotImplementedError()

    def _log_likelihood(self):
        raise NotImplementedError()

    def _dLog_likelihood(self):
        raise NotImplementedError()

    def _getParams(self):
        return self.mean.params + self.kernel.params

    def _setParams(self, params):
        assert len(params) == self.n_params
        l_mean = len(self.mean._params)
        l_kern = len(self.kernel._params)
        self.mean._params.values = params[:l_mean]
        self.kernel._params.values = params[l_mean:l_mean+l_kern]
    params = property(_getParams, _setParams)

    def __str__(self):
        ret = "Model : " + self.name + "\n"
        ret += "\tinput_dim : {}".format(self.input_dim) + "\n"
        ret += "\tn_points  : {}".format(self.n_points)
        return ret

