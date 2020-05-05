"""scipyでハイパーパラメータの最適化"""
import numpy as np
import scipy as sp
from scipy import optimize

class spOpt:

    def __init__(self, n_init=10):
        self.n_init = n_init

    def minimize(self, gp):
        """
        scipy.optimizeでハイパーパラメータの最適化

        Parameters
        ----------
        gp : 
            GPモデル

        Returns
        -------
        result : bool
            True : 最適化成功, False : 最適化失敗
        """

        if(gp.n_points == 0):
            print("Cannot optimize due to n_points=0.")
            return False

        func = gp._log_likelihood
        args = (gp.x, gp.y)
        jac = gp._dLog_likelihood

        bounds = []
        low = np.array([])
        high = np.array([])
        for mn, mx in zip(gp.params.min, gp.params.max):
            low = np.append(low, mn)
            high = np.append(high, mx)
            bounds.append((mn, mx))

        opt_val = None
        opt_params = np.zeros(shape=gp.n_params)
        init = gp.rand.uniform(low=low, high=high, size=[self.n_init, gp.n_params])
        for x0 in init:
            result = optimize.minimize(func, x0, args, method="L-BFGS-B", jac=jac, bounds=bounds)
            if result.success and (opt_val is None or result.fun < opt_val):
                opt_val = result.fun
                opt_params = result.x

        if opt_val is None:
            print("Failed optimize")
            return False

        gp.params = opt_params
        return True