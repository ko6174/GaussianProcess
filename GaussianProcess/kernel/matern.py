"""Maternカーネル."""
import numpy as np

from scipy.spatial.distance import cdist as distance

from .base import MyKernel
from ..hyper_parameter.param import Param
from ..hyper_parameter.param_list import ParamList

class Matern32(MyKernel):
    """
    Matern32カーネル.
    k(x1,x2) = variance*(1 + sqrt(3)*||x1-x2|| / lengthscale)*exp(-sqrt(3)*||x1-x2|| / lengthscale)

    Attributes
    ----------
    params : tuple
        ハイパーパラメータを格納するタプル.
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
        name = "Matern32"
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
            K = variance*(1+np.sqrt(3)*r)*np.exp(-np.sqrt(3)*r)

        return K

    def paramsGrad(self, X, params=None):
        """
        共分散行列をtheta_d(d番目のハイパーパラメータ)で微分した行列を返す関数.

        Parameters
        ----------
        X : numpy.ndarray
            入力点.
        params : list, default None
            カーネルのハイパーパラメータ. Noneのときは現在のハイパーパラメータで計算される.
        
        Returns
        -------
        K : numpy.ndarray
            dk(X1, X2) / dtheta_d
        """
        if(params is None):
            params = self.params.values
        variance = np.exp(params[0])
        lengthscale = np.exp(params[1])

        # 距離行列
        #distance_matrix = distance(X1,X2,metric="sqeuclidean")
        r = np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :])**2, axis=2)) / lengthscale

        ret = np.zeros(shape=[self.n_params, X.shape[0], X.shape[0]])
        ret[0] = variance * (1+np.sqrt(3)*r) * np.exp(-np.sqrt(3)*r)
        ret[1] = 3 * r**2 * variance * np.exp(-np.sqrt(3)*r)
        return ret

class Matern52(MyKernel):
    """
    Matern52カーネル.
    k(x1,x2) = variance*(1 + sqrt(5)*||x1-x2|| / lengthscale + 5/3*||x1-x2||^2 / lengthscale^2)*exp(-sqrt(5)*||x1-x2|| / lengthscale)

    Attributes
    ----------
    params : tuple
        ハイパーパラメータを格納するタプル.
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
        name = "Matern52"
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
            K = variance * (1+np.sqrt(5)*r + 5/3*r**2) * np.exp(-np.sqrt(5)*r)

        return K

    def paramsGrad(self, X, params=None):
        """
        共分散行列をtheta_d(d番目のハイパーパラメータ)で微分した行列を返す関数.

        Parameters
        ----------
        X : numpy.ndarray
            入力点.
        params : list, default None
            カーネルのハイパーパラメータ. Noneのときは現在のハイパーパラメータで計算される.
        
        Returns
        -------
        K : numpy.ndarray
            dk(X1, X2) / dtheta_d
        """
        if(params is None):
            params = self.params.values
        variance = np.exp(params[0])
        lengthscale = np.exp(params[1])

        # 距離行列
        #distance_matrix = distance(X1,X2,metric="sqeuclidean")
        r = np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :])**2, axis=2)) / lengthscale

        ret = np.zeros(shape=[self.n_params, X.shape[0], X.shape[0]])
        ret[0] = variance * (1+np.sqrt(5)*r + 5/3*r**2) * np.exp(-np.sqrt(5)*r)
        ret[1] = variance * r * np.exp(-np.sqrt(5)*r) * (5*np.sqrt(5)*r**2+5*r-3) / 3
        return ret
