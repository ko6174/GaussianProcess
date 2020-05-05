import numpy as np
import scipy as sp

def block_inv(Ainv, B, D):
    """
    A^{-1}が既知のときの[A, B; B^T, D]^{-1}の計算

    Parameters
    ----------
    Ainv : numpy.ndarray
        ブロック行列の左上成分の逆行列
    B : numpy.ndarray
        ブロック行列の左下成分
    D : numpy.ndarray
        ブロック行列の右下成分

    Returns
    -------
    K : numpy.ndarray
        [A, B^T; B, D]^{-1}
    """
    S = D - np.dot(np.dot(np.transpose(B), Ainv), B)
    Sinv = np.linalg.inv(S)
    ABS = np.dot(np.dot(Ainv, B), Sinv)
    CA = np.dot(np.transpose(B), Ainv)

    K11 = Ainv + np.dot(ABS, CA)
    K12 = -ABS
    K21 = -np.dot(Sinv, CA)
    K22 = Sinv

    K = np.concatenate((np.concatenate((K11, K21)), np.concatenate((K12, K22))), axis=1)
    
    return K

def block_chol(LA, B, D):
    """
    Aのコレスキー分解A = LALA^Tが既知のときの[A, B; B^T, D]のコレスキー分解の計算

    Parameters
    ----------
    LA : numpy.ndarray
        Aのコレスキー分解後の下三角行列
    B : numpy.ndarray
        ブロック行列の右上成分
    D : numpy.ndarray
        ブロック行列の右下成分

    Returns
    -------
    Lk : numpy.ndarray
        [A, B; B^T, D]をコレスキー分解したときの下三角行列
    """
    if(len(LA) == 0):
        return np.linalg.cholesky(D)
    
    x = sp.linalg.solve_triangular(LA, B, lower=True)
    S = D - np.dot(np.transpose(x), x)
    LS = np.linalg.cholesky(S)
    Z = np.zeros(shape=[LA.shape[0], D.shape[0]], dtype=np.float)
    Lk = np.concatenate((np.concatenate((LA, np.transpose(x))), np.concatenate((Z, LS))), axis=1)
    return Lk

def forward(L, b):
    """
    下三角行列Lと行列bに対してL^{-1}bを計算.

    L : numpy.ndarray
        下三角行列.
    b : numpy.ndarray
        L^{-1}と内積をとる行列.
    """
    if(len(L) == 0):
        return np.array([[0.]])
    
    return sp.linalg.solve_triangular(L, b, lower=True)

def dot_matinv_vec(L, b, quad=False):
    """
    A = LL^Tのとき, bA^{-1}bまたはA^{-1}bを計算.

    Parameters
    ----------
    L : numpy.ndarray
        行列Aをコレスキー分解したときの下三角行列.
    b : numpy.ndarray
        A^{-1}との内積をとる行列.
    quad : bool
        bA^{-1}bとA^{-1}bのどちらを計算するか.
        TrueでbA^{-1}b, FalseでA^{-1}bを計算.
    
    Returns
    -------
    ret : numpy.ndarray
        quad ? bA^{-1}b : A^{-1}b
    """
    ret = sp.linalg.solve_triangular(L, b, lower=True)
    if quad:
        ret = ret.transpose().dot(ret)
    else:
        ret = sp.linalg.solve_triangular(np.transpose(L), ret)

    return ret


def triangular_inv(L):
    """
    下三角行列Lの逆行列の計算

    Parameters
    ----------
    L : numpy.ndarray
        下三角行列

    Returns
    -------
    Linv : numpy.ndarray
        Lの逆行列
    """
    n = L.shape[0]
    Linv = np.diag(1/np.diag(L))

    for j in range(n-1):
        for i in range(j+1, n):
            Linv[i,j] = -np.dot(L[i, j:i], Linv[j:i, j]) / L[i,i]
    
    return Linv

def block_triangular_inv(Linv, B, D):
    """
    下三角行列Lc = [L, 0; B, D]のLinvが既知のときのLcinvの計算

    Parameters
    ----------
    Linv : numpy.ndarray
        Lの逆行列
    B : numpy.ndarray
        ブロック下三角行列の左下成分
    D : numpy.ndarray
        ブロック下三角行列の右下成分

    Returns
    -------
    Lcinv : numpy.ndarray
        [L, 0; B, D]の逆行列
    """
    Dinv = triangular_inv(D)
    B = -np.dot(np.dot(Dinv, B), Linv)
    Z = np.zeros(shape=[Linv.shape[0], D.shape[0]])

    Lcinv = np.concatenate((np.concatenate((Linv, B)), np.concatenate((Z, Dinv))), axis=1)
    
    return Lcinv

def solve_forward(L, B):
    """
    A = LL^Tとして, np.dot(A^{-1}, B)を計算

    Parameters
    ----------
    L : numpy.ndarray
        Aのコレスキー分解したときの下三角行列
    B : numpy.ndarray
        A^{-1}に掛ける行列
    
    Returns
    -------
    np.dot(A^{-1}, B)
    """
    pass
