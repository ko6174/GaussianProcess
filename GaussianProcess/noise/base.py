"""ノイズのテンプレートと実装."""
from abc import ABCMeta, abstractmethod

class MyNoise(metaclass=ABCMeta):

    def __init__(self, params, name):
        self._name = name
        self._params = params
        self.n_params = len(params)

    @abstractmethod
    def getNoise(self, X, params=None):
        pass

    @abstractmethod
    def paramsGrad(self):
        pass

    def _getParams(self):
        return self._params
    
    def _setParams(self, values):
        self._params.values = values
    params = property(_getParams, _setParams)

    def _getName(self):
        return self._name
    name = property(_getName)
    
    def __call__(self, X, params=None):
        return self.getNoise(X, params)

    def __str__(self):
        ret = "Noise : " + self.name + "\n"
        ret += self.params.__str__() + "\n"

        return ret
