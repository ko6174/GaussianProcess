"""定数を返す事前平均関数."""
import numpy as np
from .base import MyMean
from ..hyper_parameter.param import Param
from ..hyper_parameter.param_list import ParamList

class Zero(MyMean):
    """
    定数(0)関数.

    Attributes
    ----------
    params : list
        ハイパーパラメータのリスト.
    """

    def __init__(self):
        """
        コンストラクタ.
        """
        params = ParamList()
        name = "Zero"
        super().__init__(params, name)

    def getMean(self, X, params=None):
        n, _ = X.shape
        return np.zeros(shape=[n,1])
    
    def paramsGrad(self, X, params=None):
        pass