import numpy as np
cimport numpy as np
import scipy as sp
import cython
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck( False )
@cython.wraparound( False )
@cython.nonecheck( False )
def IRLS( np.ndarray[DTYPE_t, ndim=2] K, np.ndarray[DTYPE_t, ndim = 2] y , int max_iter=1000, float eps=1e-5):

    cdef int N = K.shape[0]
    cdef float log_likelihood, prev_likelihood=1e6

    cdef np.ndarray[DTYPE_t, ndim=2] f = np.zeros(shape=[N,1], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] t = (y + 1) / 2, pi, b, a
    cdef np.ndarray[DTYPE_t, ndim=2] W, L, WL, LWKb

    for i in xrange(max_iter):
        pi = 1 / (1 + np.exp(f))
        W = np.diag(np.sqrt((pi * (1-pi))[:,0]))
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
