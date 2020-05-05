"""グリッドサーチでハイパーパラメータの最適化"""
import numpy as np
import scipy as sp

class GridSearch:

    def __init__(self, cand_params=None):
        self.cand_params = cand_params

    def minimize(self, gp):
        """
        グリッドでハイパーパラメータの最適化

        Parameters
        ----------
        gp : 
            GPモデル
        cand_params : list, default None
            パラメータ候補

        Returns
        -------
        result : bool
            True : 最適化成功, False : 最適化失敗
        """

        if(gp.n_points == 0):
            print("Cannot optimize due to n_points=0.")
            return False

        cand_params = self.cand_params
        if cand_params is None:
            cand = np.meshgrid(*np.linspace(gp.params.min, gp.params.max, 5).transpose())
            cand_params = np.array(cand).transpose().reshape(-1, gp.n_params)

        func = gp._log_likelihood

        opt_val = None
        opt_params = np.zeros(shape=gp.n_params)
        for params in cand_params:
            try:
                fun = func(params, gp.x, gp.y)
                result = True
            except:
                result = False

            if result and (opt_val is None or fun < opt_val):
                opt_val = fun
                opt_params = params

        if opt_val is None:
            print("Failed optimize")
            return False

        gp.params = opt_params
        return True