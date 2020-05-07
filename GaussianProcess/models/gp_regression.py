"""ガウス過程回帰の実装."""
import numpy as np
import scipy as sp
from scipy import optimize

from .base import MyModel
from ..kernel import RBF
from ..mean import Const
from ..noise import Gauss
from ..optimizer import spOpt
from ..hyper_parameter import Param, ParamList
from ..lib import block_chol, forward, dot_matinv_vec

class GPRegression(MyModel):
    """
    ガウス過程回帰

    Attributes
    ----------
    mean : MyMean
        事前平均関数.
    kernel : MyKernel
        カーネル関数.
    noise : MyNoise
        ノイズ.
    input_dim : int
        入力の次元.
    params : list
        ハイパーパラメータのリスト.
    rand : numpy.random.RandomState
        乱数生成器.
    n_points : int
        観測点数.
    x : numpy.ndarray
        観測値に対応する入力点.
    y : numpy.ndarray
        観測値.
    Lk : numpy.ndarray
        共分散行列をコレスキー分解したときの下三角行列.
    """

    def __init__(self, mean=None, kernel=None, input_dim=1, noisevariance=None, seed=None):
        """
        コンストラクタ.

        Parameters
        ----------
        mean : Mymean, default Const()
            事前平均関数. デフォルトは0を返す定数関数.
        kernel : Mykernel, default RBF()
            カーネル関数. デフォルトはRBFカーネル.
        input_dim : int, default 1
            入力の次元.
        noisevariance : float, default 0
            観測ノイズの分散.
        seed : int, default None
            乱数生成器のseed値.
        """

        name = "Gaussian Process Regression"
        self.noise = Gauss(np.log(noisevariance)) if noisevariance is not None else Gauss()
        super().__init__(name, mean, kernel, input_dim, seed)

        self.n_params += self.noise.n_params
        self.x = np.empty(shape=[0,input_dim])
        self.y = np.empty(shape=[0,1])
        self.Lk = np.empty(shape=[0,0])

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

        K_xX = self.kernel(self.x, X)
        K_XX = self.kernel(X, X) + self.noise(X)
        self.Lk = block_chol(self.Lk, K_xX, K_XX)
        
        self.x = np.concatenate((self.x, X))
        self.y = np.concatenate((self.y, Y))
        self.n_points += n

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
        y, y_mean, y_std = self.normalize(self.y - self.mean(self.x))

        K_xX = self.kernel(self.x, X)
        alpha = np.transpose(dot_matinv_vec(self.Lk, K_xX))
        y_mean = alpha.dot(y_mean * np.ones(shape=[self.n_points,1]))

        mean = self.mean(X) + alpha.dot(y) * y_std + y_mean
        if diag:
            var = self.kernel(X, X, diag=diag) - np.sum(alpha.transpose() * K_xX, axis=0)
            var = np.reshape(var, [-1,1])
        else:
            var = self.kernel(X, X, diag=diag) - alpha.dot(K_xX)

        return mean, var

    def sampling(self, X, n_samples=1):
        """
        関数のサンプリング

        Parameters
        ----------
        X : numpy.ndarray
            入力点
        n_samples : int, default 1
            サンプル数

        Returns
        -------
        samples : numpy.ndarray
            Xにおけるサンプル値
        """
        mean, Cov = self.predict(X, diag=False)
        return self.rand.multivariate_normal(mean[:,0], Cov, size=n_samples)

    def optimize(self, optim=None):
        """
        ハイパーパラメータチューニング.

        Parameters
        ----------
        optim :
            最適化クラス. デフォルトはscipy.optimize.minimize

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

    def normalize(self, y):
        """
        yを平均0, 分散1に標準化.

        Parameters
        ----------
        y : numpy.ndarray
            標準化したい変数

        Returns
        -------
        ret : numpy.ndarray
            yを標準化したもの
        mean : numpy.ndarray
            yの平均
        std : numpy.ndarray
            yの標準偏差
        """
        mean = np.mean(y)
        std = np.std(y)
        ret = (y - mean) / (std + 1e-7)
        return ret, mean, std

    def model_update(self):
        """
        現在のハイパーパラメータでグラム行列を更新. ハイパーパラメータの変更後に使用.
        """
        self.Lk = np.linalg.cholesky(self.kernel(self.x, self.x) + np.exp(self.noise.params.values[0])*np.identity(self.n_points))
        

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
        n, _ = X.shape
        l_mean = self.mean.n_params
        l_kern = self.kernel.n_params
        Cov = self.kernel(X, X, params[l_mean:l_mean+l_kern]) + self.noise(X, params=params[l_mean+l_kern:])
        Lk = np.linalg.cholesky(Cov)
        y, _, _ = self.normalize(Y - self.mean(X, params[:l_mean]))
        Linvy = forward(Lk, y)

        log_likelihood = -np.sum(np.log(np.diag(Lk))) - 0.5*np.sum(Linvy**2) - 0.5*n*np.log(2*np.pi)

        return -log_likelihood

    def _dLog_likelihood(self, params, X, Y):
        """
        -対数周辺尤度の微分の計算.

        Parameters
        ----------
        params : list
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
        l_mean = self.mean.n_params
        l_kern = self.kernel.n_params

        Cov = self.kernel(X, X, params[l_mean:l_mean+l_kern]) + self.noise(X, params=params[l_mean+l_kern:])
        Lk = np.linalg.cholesky(Cov)
        y, _, _ = self.normalize(Y - self.mean(X, params[:l_mean]))

        Kinvy = dot_matinv_vec(Lk, y)

        dL = np.array([], dtype=np.float)

        if self.mean.n_params > 0:
            paramsGrad = np.transpose(self.mean.paramsGrad(X, params=params[:l_mean]), axes=[0,2,1])
            dL = np.append(dL, -2*np.dot(paramsGrad, Kinvy)[:,0,0])

        if self.kernel.n_params > 0:
            paramsGrad = self.kernel.paramsGrad(X, params=params[l_mean:l_mean+l_kern])
            for dK in paramsGrad:
                tr = np.trace(dot_matinv_vec(Lk, dK))
                dL = np.append(dL, tr - np.dot(np.transpose(Kinvy), np.dot(dK, Kinvy))[0,0])

        if self.noise.n_params > 0:
            paramsGrad = self.noise.paramsGrad(X, params=params[l_mean+l_kern:])
            for dK in paramsGrad:
                tr = np.trace(dot_matinv_vec(Lk, dK))
                dL = np.append(dL, tr - np.dot(np.transpose(Kinvy), np.dot(dK, Kinvy))[0,0])

        return dL

    def _getParams(self):
        return self.mean._params + self.kernel._params + self.noise._params

    def _setParams(self, params):
        assert len(params) == self.n_params
        l_mean = len(self.mean._params)
        l_kern = len(self.kernel._params)
        self.mean._params.values = params[:l_mean]
        self.kernel._params.values = params[l_mean:l_mean+l_kern]
        self.noise._params.values = params[l_mean+l_kern:]
    params = property(_getParams, _setParams)

    def __call__(self, X):
        return self.predict(X)

    def __str__(self):
        ret = super().__str__()
        ret += "\n\n"
        ret += self.mean.__str__() + "\n\n"
        ret += self.kernel.__str__() + "\n\n"
        ret += self.noise.__str__()
        return ret