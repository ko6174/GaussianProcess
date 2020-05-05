"""定数を返す事前平均関数."""
import numpy as np
from .base import MyMean
from ..hyper_parameter.param import Param
from ..hyper_parameter.param_list import ParamList

class Const(MyMean):
    """
    定数関数.

    Attributes
    ----------
    params : list
        ハイパーパラメータのリスト.
    """

    def __init__(self, const=0.):
        """
        コンストラクタ.

        Parameters
        ----------
        const : float, default 0.
            返す値.
        """
        params = ParamList([Param(const, name="constant", min=-10, max=10, log_scale=False)])
        name = "Constant"
        super().__init__(params, name)

    def getMean(self, X, params=None):
        if(params is None):
            params = self._params.values
        const = params[0]
        n, _ = X.shape
        return np.reshape(np.array([const] * n), [n,1])
    
    def paramsGrad(self, X, params=None):
        if(params is None):
            params = self._params.values
        ret = np.zeros(shape=[self.n_params, X.shape[0], 1])
        ret[0] = np.ones(shape=[X.shape[0],1])
        return ret