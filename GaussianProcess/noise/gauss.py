"""正規分布に従うノイズ."""
import numpy as np

from .base import MyNoise
from ..hyper_parameter.param import Param
from ..hyper_parameter.param_list import ParamList

class Gauss(MyNoise):
    """
    正規分布に従うノイズ.

    Attributes
    ----------
    params : tuple
        ハイパーパラメータを格納するタプル.
    n_params : int
        ハイパーパラメータの数.
    """

    def __init__(self, ln_variance=0.):
        """
        コンストラクタ.

        Parameters
        ----------
        ln_variance : float, default 0.
            log(variance)
        """
        params = ParamList([Param(ln_variance, "ln_variance", -4*np.log(10), 1*np.log(10))])
        name = "Gauss"
        super().__init__(params, name)

    def getNoise(self, X, params=None):
        """
        ノイズの分散を返す関数.

        Parameters
        ----------
        X : numpy.ndarray
            入力点.
        params : list, default None
            ノイズの分散. Noneのときは現在のハイパーパラメータで計算される.

        Returns
        -------
        noise : numpy.ndarray
            ノイズの分散の対角行列.
        """
        if(params is None):
            variance = np.exp(self._params[0].value)
        else:
            variance = np.exp(params[0])
        
        noise = variance * np.identity(X.shape[0])
        
        return noise
    
    def paramsGrad(self, X, params=None):
        """
        ノイズの分散を微分した行列を返す関数.

        Parameters
        ----------
        X : numpy.ndarray
            入力点.
        params : list, default None
            ノイズのハイパーパラメータ. Noneのときは現在のハイパーパラメータで計算される.
        
        Returns
        -------
        noise : numpy.ndarray
            微分後の行列.
        """
        if(params is None):
            variance = np.exp(self._params[0].value)
        else:
            variance = np.exp(params[0])
        
        noise = np.reshape(variance * np.identity(X.shape[0]), [1, X.shape[0], X.shape[0]])
        return noise
