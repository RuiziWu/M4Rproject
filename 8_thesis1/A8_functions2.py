import numpy as np
from tqdm import tqdm

def score_normal(X):
    
    # Sq(x[i]) = q'(x[i]) / q(x[i]) = -x[i]
    return -X


def diff_fn(X, Y):
    """
    X: m x N
    Y: m x N
    """
    
    X = np.expand_dims(X, axis=1) # m x 1 x N
    Y = np.expand_dims(Y, axis=0) # 1 x m x N
    diff = X - Y # m x m x N
    return diff


def diff_norm_sq_fn(X, Y):
    """
    X: m x N
    Y: m x N
    """
    
    diff = diff_fn(X, Y) # m x m x N
    diff_norm_sq = np.sum(diff**2, axis=-1) # m x m
    return diff_norm_sq


def kernel_function(X, h):
    """
    Args:
        X: m x N
        h: float
    """

    m, N = X.shape

    diff_norm_sq = diff_norm_sq_fn(X, X) # m x m
    kernelMatrix = np.exp(- 1 / (2 * h**2) * diff_norm_sq) # m x m

    kernelMatrix_expand  = np.expand_dims(kernelMatrix, axis=-1) # m x m x 1
    diff = diff_fn(X, X) # m x m x N
    gradKernel1 = - 1 / (h**2) * diff * kernelMatrix_expand # m x m x N
    gradKernel2 = - gradKernel1 # m x m x N

    hessKernel = (N - (1 / h **2) * diff_norm_sq) * kernelMatrix / h ** 2
    
    return kernelMatrix, gradKernel1, gradKernel2, hessKernel


def UqMatrix(X, h):

    kernelMatrix, gradKernel1, gradKernel2, hessKernel = kernel_function(X, h)
    X_expand = np.expand_dims(-X, axis = 1) # m x 1 x N
    Y_expand = np.expand_dims(-X, axis = 0) # 1 x m x N
    UMatrix = kernelMatrix * np.dot(X, X.T) + np.sum(X_expand * gradKernel2, axis = -1) + np.sum(Y_expand * gradKernel1, axis = -1) + hessKernel

    return UMatrix


def KSD(U):

    m, _ = U.shape
    matDiag = np.sum(U.diagonal())
    matSum = U.sum()
    KSD = (matSum - matDiag) / (m * (m - 1))
    
    return KSD


def Bootstrap_KSD(U, size = 1000):
    """
    
    """

    m, _ = U.shape
    multi_prob = np.repeat((1 / m), m)

    Sstar = np.zeros(size)
    for i in range(size):
        Weight = np.random.multinomial(m, multi_prob)
        Wadjust = (Weight - 1) / m
        WMatrix = np.outer(Wadjust, Wadjust)
        SMatrix = WMatrix * U
        diag_sum = sum(SMatrix.diagonal())
        matrix_sum = SMatrix.sum()
        Si = matrix_sum - diag_sum
        Sstar[i] = Si

    return Sstar


def approx_pvalue(S, Sstar):
    """
    param S: unbiased estimation of KSD, scalar
    param Sstar: unbiased m bootstrap sample KSD
    """
    n = len(Sstar)
    TFarray = Sstar[Sstar >= S]
    count = len(TFarray)
    return count / n


def pValue_allmeanshift(samplesize, dim_list, meanvalue, bandwidth, bootstrapsize = 1000, iter = 100):
    """
    param stepvalue: 1D numpy array with dimension dim or boolean value False
    param covalue: 1D numpy array with dimension dim or boolean value False

    param
    """
    h = bandwidth
    m = meanvalue
    n = len(dim_list)
    pvalue = np.zeros((n, iter))
    
    for i in tqdm(range(n)):
        dim = dim_list[i]
        cov = np.identity(dim)
        mean = np.repeat(m, dim)
        for j in tqdm(range(iter)):
            Multinormal_X = np.random.multivariate_normal(mean, cov, samplesize)
            UMatrix = UqMatrix(Multinormal_X, h)
            KSDvalue = KSD(UMatrix)
            KSDstar = Bootstrap_KSD(UMatrix, size = bootstrapsize)
            pvalue[i, j] = approx_pvalue(KSDvalue, KSDstar)
    return pvalue


def pDist_H0_dimchange(samplesize, dim_list, bandwidth, bootstrapsize = 1000, iter = 100):
    """
    dim: a list of dimensions
    """
    h = bandwidth
    n = len(dim_list)
    pvalue = np.zeros((n, iter))
    
    for i in tqdm(range(n)):
        dim = dim_list[i]
        mean = np.zeros(dim)
        cov = np.identity(dim)
        for j in tqdm(range(iter)):
            Multinormal_X = np.random.multivariate_normal(mean, cov, samplesize)
            UMatrix = UqMatrix(Multinormal_X, h)
            KSDvalue = KSD(UMatrix)
            KSDstar = Bootstrap_KSD(UMatrix, size = bootstrapsize)
            pvalue[i, j] = approx_pvalue(KSDvalue, KSDstar)

    return pvalue


def pDist_H1_dimchange(samplesize, dim_list, meanshift, bandwidth, bootstrapsize = 1000, iter = 100):
    """
    dim: a list of dimensions
    """
    h = bandwidth
    n = len(dim_list)
    pvalue = np.zeros((n, iter))
    
    for i in tqdm(range(n)):
        dim = dim_list[i]
        mean = np.zeros(dim)
        mean[0] = meanshift
        cov = np.identity(dim)
        for j in tqdm(range(iter)):
            Multinormal_X = np.random.multivariate_normal(mean, cov, samplesize)
            UMatrix = UqMatrix(Multinormal_X, h)
            KSDvalue = KSD(UMatrix)
            KSDstar = Bootstrap_KSD(UMatrix, size = bootstrapsize)
            pvalue[i, j] = approx_pvalue(KSDvalue, KSDstar)

    return pvalue