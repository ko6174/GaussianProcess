"""GP binary classification"""
import numpy as np
import scipy as sp

from .base import MyModel
from ..optimizer import spOpt
from ..lib import IRLS, forward

class GPBiClassification(MyModel):
    """
    GPの二値分類.
    """

    def __init__(self, kernel=None, input_dim=1, seed=None):
        """
        コンストラクタ.

        Parameters
        ----------
        kernel : Mykernel, default RBF()
            カーネル関数. デフォルトはRBFカーネル.
        input_dim : int, default 1
            入力の次元.
        ln_noisevariance : float, default 0
            観測ノイズの分散のlogスケール.
        seed : int, default None
            乱数生成器のseed値.
        """
        name = "Gaussian Process Binary Classification"
        mean = None
        super().__init__(name, mean, kernel, input_dim, seed)

        self.n_params = self.kernel.n_params
        self.x = np.empty(shape=[0,input_dim])
        self.y = np.empty(shape=[0,1])
        self.f = None
    
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
        n, _ = X.shape

        self.x = np.concatenate((self.x, X))
        self.y = np.concatenate((self.y, Y))
        self.n_points += n
        self.f = None

    def predict(self, X, diag=True, restart=False):
        """
        Xにおける予測平均, 予測分散の計算.

        Parameters
        ----------
        X : numpy.ndarray
            入力点.
        diag : bool, default True
            共分散行列の対角成分のみを計算するかどうか.
        restart : bool, default False
            fを再計算するかどうか.

        Returns
        -------
        mean : np.ndarray
            予測平均.
        var : np.ndarray
            予測分散.
        """
        n, _ = X.shape

        if self.n_points == 0:
            mean = np.zeros(shape=[n,1])
            var = self.kernel(X, X, diag=diag)
            if diag:
                var = var.reshape(n,1)
        else:
            if self.f is None or restart:
                self.f, _ = IRLS(self.kernel(self.x, self.x), self.y)

            t = (self.y + 1) / 2
            pi = 1 / (1 + np.exp(-self.f))
            k = self.kernel(self.x,X)
            mean = k.transpose().dot(t-pi)

            K = self.kernel(self.x, self.x)
            W = np.diag(np.sqrt(pi * (1-pi))[:,0])
            L = np.linalg.cholesky( np.identity(self.n_points) + W.dot(K.dot(W)) )

            v = forward(L, W.dot(k))

            if diag:
                var = self.kernel(X, X, diag=diag) - np.sum(v * v, axis=0)
                var = np.reshape(var, [-1,1])
            else:
                var = self.kernel(X, X, diag=diag) - v.transpose().dot(v)

        return mean, var

    def predict_prob(self, X, restart=False):
        """
        Xにおけるy=1となる確率の計算.

        Parameters
        ----------
        X : numpy.ndarray
            入力点.
        restart : bool, default False
            fを再計算するかどうか.

        Returns
        -------
        prob : np.ndarray
            y=1となる確率.
        var : np.ndarray
            予測分散.
        """
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        mean, var = self.predict(X, restart=restart)
        kappa = 1 / np.sqrt(1 + np.pi * var / 8)
        prob = sigmoid(kappa * mean)

        return prob, var

    def sampling(self, X, n_samples=1, restart=False):
        """
        関数のサンプリング.

        Parameters
        ----------
        X : numpy.ndarray
            入力点.
        n_samples : int, default 1
            サンプル数.
        restart : bool, default False
            fを再計算するかどうか.

        Returns
        -------
        samples : numpy.ndarray
            Xにおけるサンプル値.
        """
        mean, Cov = self.predict(X, diag=False, restart=restart)
        return self.rand.multivariate_normal(mean[:,0], Cov, size=n_samples)

    def convert_prob(self, f):
        """
        fから確率への変換.

        Parameters
        ----------
        f : numpy.ndarray
            関数値.

        Returns
        -------
        prob : numpy.ndarray
            fを確率に変換した値.
        """
        return 1 / (1 + np.exp(-f))

    def optimize(self, optim=None):
        """
        ハイパーパラメータチューニング.

        Parameters
        ----------
        optim :
            最適化クラス. デフォルトはscipy.optimize.minimize
        n_init : int, default 10
            初期点の数

        Returns
        -------
        result.success : bool
            最適化が成功したかどうか.
        """
        if optim is None:
            optim = spOpt()
        result = optim.minimize(self)
        self.model_update()
        return result

    def model_update(self):
        self.f = None

    def _log_likelihood(self, params, X, Y):
        """
        -対数周辺尤度の計算.

        Parameters
        ----------
        params : numpy.ndarray
            ハイパーパラメータのリスト.
        X : numpy.ndarray
            尤度に対応する入力点.
        Y : numpy.ndarray
            Xに対応する観測値.

        Returns
        -------
        -log_likelihood : numpy.ndarray
            対数周辺尤度に-1を掛けた値.
        """
        K = self.kernel(X, X, params=params)
        _, ret = IRLS(K, Y)
        return -ret

    def _dLog_likelihood(self, params, X, Y):
        """
        -対数周辺尤度の微分の計算.

        Parameters
        ----------
        params : numpy.ndarray
            ハイパーパラメータのリスト.
        X : numpy.ndarray
            尤度に対応する入力点.
        Y : numpy.ndarray
            Xに対応する観測値.

        Returns
        -------
        dL : numpy.ndarray
            -対数周辺尤度をハイパーパラメータベクトルで微分したもの.
        """
        n, _ = X.shape
        K = self.kernel(X, X, params=params) + 1e-8 * np.identity(n)
        f, _ = IRLS(K, Y)

        t = (Y + 1) / 2
        pi = 1 / (1 + np.exp(-f))
        W = np.diag(np.sqrt(pi * (1-pi))[:,0])
        LB = np.linalg.cholesky( np.identity(n) + W.dot(K.dot(W)) )
        b = (W*W).dot(f) + t - pi
        WL = forward(LB, W)
        WL = WL.transpose()
        LWKb = forward(LB, W.dot(K).dot(b))
        a = b - WL.dot(LWKb)

        sqrtR = forward(LB, W)
        R = sqrtR.transpose().dot(sqrtR)
        C = forward(LB, W.dot(K))
        s2 = 0.5 * np.diag(K - np.sum(C * C, axis=0)) * pi*(1-pi)*(1-2*pi)

        dL = np.array([], dtype=np.float)
        if self.kernel.n_params > 0:
            paramsGrad = self.kernel.paramsGrad(X, params=params)
            for dK in paramsGrad:
                s1 = 0.5 * a.transpose().dot(dK.dot(a))[0,0] - 0.5 * np.sum(R.transpose() * C)
                b = dK.dot(t - pi)
                s3 = b = K.dot(R.dot(b))
                dL = np.append(dL, -s1 - s2.transpose().dot(s3)[0,0])

        return dL

    def _getParams(self):
        return self.kernel._params

    def _setParams(self, params):
        assert len(params) == self.n_params
        self.kernel._params.values = params
    params = property(_getParams, _setParams)

    def __call__(self, X):
        return self.predict(X)

    def __str__(self):
        ret = super().__str__() + "\n\n"
        ret += self.kernel.__str__()
        return ret