"""RBFカーネル"""
import numpy as np

from scipy.spatial.distance import cdist as distance

from .base import MyKernel
from ..hyper_parameter.param import Param
from ..hyper_parameter.param_list import ParamList

class RBF(MyKernel):
    """
    RBFカーネル(SEカーネル, ガウスカーネル).
    k(x1,x2) = variance * exp(-||x1-x2||^2 / lengthscale^2)

    Attributes
    ----------
    params : list
        ハイパーパラメータを格納するタプル.
    n_params : int
        ハイパーパラメータの数.
    name : str
        カーネルの名前.
    """

    def __init__(self, variance=1., lengthscale=1.):
        """
        コンストラクタ.

        Parameters
        ----------
        variance : float, default 1.
        lengthscale : float, default 1.
        """
        
        params = ParamList([Param(np.log(variance), "ln_variance", -1*np.log(10), 1*np.log(10)), 
                                Param(np.log(lengthscale), "ln_lengthscale", -4*np.log(10), 1*np.log(10))])
        name = "RBF"
        super().__init__(params, name)

    def getKernel(self, X1, X2, params=None, diag=False):
        """
        共分散行列を返す関数.

        Parameters
        ----------
        X1 : numpy.ndarray
            カーネルの第一引数.
        X2 : numpy.ndarray
            カーネルの第二引数.
        params : list, default None
            カーネルのハイパーパラメータ. Noneのときは現在のハイパーパラメータで計算される.
        diag : bool, default False
            対角成分のみを計算するかどうか

        Returns
        -------
        K : numpy.ndarray
            k(X1, X2)
        """
        if(params is None):
            params = self.params.values
        variance = np.exp(params[0])
        lengthscale = np.exp(params[1])

        if diag:
            K = variance * np.ones(shape=X1.shape[0])
        else:
            # 距離行列
            r = np.sqrt(np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :])**2, axis=2)) / lengthscale
            K = variance * np.exp(-r**2)

        return K
    
    def paramsGrad(self, X, params=None):
        """
        共分散行列の微分を返す関数.

        Parameters
        ----------
        X : numpy.ndarray
            カーネルの第一引数.
        params : list, default None
            カーネルのハイパーパラメータ. Noneのときは現在のハイパーパラメータで計算される.
        
        Returns
        -------
        ret : numpy.ndarray
            dk(X, X) / dtheta_d
        """
        if(params is None):
            params = self.params.values
        variance = np.exp(params[0])
        lengthscale = np.exp(params[1])

        # 距離行列
        #distance_matrix = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :])**2, axis=2)
        r = np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :])**2, axis=2)) / lengthscale

        ret = np.zeros(shape=[self.n_params, X.shape[0], X.shape[0]])
        ret[0] = variance * np.exp(-r**2)
        ret[1] = variance * np.exp(-r**2) * 2*r**2
        return ret