import numpy as np
import scipy as sp

def IRLS(K, y, max_iter=1000, eps=1e-7):

    N = K.shape[0]
    prev_likelihood=1e6

    f = np.zeros(shape=[N,1])
    t = (y + 1) / 2

    for _ in range(max_iter):
        pi = 1 / (1 + np.exp(-f))
        W = np.diag(np.sqrt(pi * (1-pi))[:,0])
        L = np.linalg.cholesky( np.identity(N) + W.dot(K.dot(W)) )
        b = (W*W).dot(f) + t - pi
        WL = sp.linalg.solve_triangular(L, W, lower=True)
        WL = WL.transpose()
        LWKb = sp.linalg.solve_triangular(L, W.dot(K).dot(b), lower=True)
        a = b - WL.dot(LWKb)
        f = K.dot(a)
        log_likelihood = np.sum(np.log(pi)) - 0.5*a.transpose().dot(f)[0,0] - np.sum(np.log(np.diag(L)))
        if np.abs(log_likelihood - prev_likelihood) < eps:
            break
        prev_likelihood = log_likelihood

    return f, log_likelihood
