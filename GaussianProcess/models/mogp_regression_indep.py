"""Multiple-Output GP Regression"""
import numpy as np
import scipy as sp
from scipy import optimize

from . import GPRegression

class MOGPR_indep:
    """
    多出力のガウス過程回帰
    各出力間は独立を仮定
    """

    def __init__(self, n_func=2, mean=None, kernel=None, input_dim=1, ln_noisevariance=None, seed=None):
        """
        コンストラクタ

        Parameters
        ----------
        n_func : int, default 2
            目的関数の数
        mean : Mymean, default None
            事前平均関数. デフォルトは0を返す定数関数
        kernel : Mykernel, default None
            カーネル関数. デフォルトはRBFカーネル
        input_dim : int, default 1
            入力の次元.
        ln_noisevariance : float, default 0
            観測ノイズの分散のlogスケール
        seed : int, default None
            乱数生成器のseed値.
        """
        self.n_func = n_func
        if mean is None:
            mean = [None]*n_func
        if kernel is None:
            kernel = [None]*n_func
        if ln_noisevariance is None:
            ln_noisevariance = [None]*n_func

        self.so_gp = [GPRegression(mean[i], kernel[i], input_dim, ln_noisevariance[i], seed) for i in range(n_func)]
        self.n_params = 0
        for gp in self.so_gp:
            self.n_params += gp.n_params
        
        self.name = "Multiple-Output Gaussian Process Regression (independent)"

    def add_points(self, X, Y):
        """
        観測点の追加.

        Parameters
        ----------
        X : numpy.ndarray
            入力点.
        Y : numpy.ndarray
            観測値.
        """
        for i in range(self.n_func):
            self.so_gp[i].add_points(X, Y[:,[i]])

    def predict(self, X, diag=True):
        """
        Xにおける予測平均, 予測分散の計算.

        Parameters
        ----------
        X : numpy.ndarray
            入力点.
        diag : bool, default True
            共分散行列の対角成分のみを計算するかどうか

        Returns
        -------
        mean : np.ndarray
            予測平均.
        var : np.ndarray
            予測分散.
        """
        n, _ = X.shape
        mean = np.zeros(shape=[n, self.n_func])

        if diag:
            var = np.zeros(shape=[n, self.n_func])

            for i in range(self.n_func):
                m, v = self.so_gp[i].predict(X, diag)
                mean[:,[i]] = m
                var[:,[i]] = v
        else:
            var = np.zeros(shape=[self.n_func, n, n])

            for i in range(self.n_func):
                m, v = self.so_gp[i].predict(X, diag)
                mean[:,[i]] = m
                var[i] = v
        
        return mean, var

    def sampling(self, X, n_samples=1):
        """
        関数のサンプリング.

        Parameters
        ----------
        X : numpy.ndarray
            入力点.
        
        n_samples : int, default 1
            サンプル数.

        Returns
        -------
        samples : numpy.ndarray
            Xにおけるサンプル値.
        """
        n, _ = X.shape
        samples = np.zeros(shape=[self.n_func, n_samples, n])

        for i in range(self.n_func):
            sample = self.so_gp[i].sampling(X)
            samples[i] = sample
        
        return samples

    def optimize(self, optim=None):
        """
        ハイパーパラメータチューニング. scipy.optimize.minimizeを使用.

        Parameters
        ----------
        optim :
            最適化クラス.

        Returns
        -------
        result.success : bool
            最適化が成功したかどうか.
        """

        for i in range(self.n_func):
            self.so_gp[i].optimize(optim)

    def __str__(self):
        ret = "Model : " + self.name + "\n"
        ret += "\tinput_dim : {}".format(self.so_gp[0].input_dim) + "\n"
        ret += "\tn_points  : {}".format(self.so_gp[0].n_points) + "\n\n"

        for i in range(self.n_func):
            ret += "=======================================================================\n"
            ret += "model {}".format(i+1) + "\n"
            ret += self.so_gp[i].__str__() + "\n\n"

        return ret