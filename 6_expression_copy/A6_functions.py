import numpy as np
import math
from tqdm import tqdm

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


def kernel_function(X, set_bandwidth = True):
    """
    Args:
        X: m x N
        h: float
    """

    m, N = X.shape

    diff_norm_sq = diff_norm_sq_fn(X, X) # m x m
    if set_bandwidth == True:
        h = np.sqrt(N / 2)

    else:
        h = set_bandwidth
        
    kernelMatrix = np.exp(- 1 / (2 * h**2) * diff_norm_sq) # m x m

    kernelMatrix_expand  = np.expand_dims(kernelMatrix, axis=-1) # m x m x 1
    diff = diff_fn(X, X) # m x m x N
    gradKernel1 = - 1 / (h**2) * diff * kernelMatrix_expand # m x m x N
    gradKernel2 = - gradKernel1 # m x m x N

    hessKernel = (N - (1 / h **2) * diff_norm_sq) * kernelMatrix / h ** 2
    
    return kernelMatrix, gradKernel1, gradKernel2, hessKernel


def UqMatrix(X, mean, set_bandwidth = True):
    m, _ = X.shape
    Mean = np.repeat(mean, m).reshape(-1, m).T
    
    kernelMatrix, gradKernel1, gradKernel2, hessKernel = kernel_function(X, set_bandwidth = set_bandwidth)
    X_expand = np.expand_dims(-(X - Mean), axis = 1) # m x 1 x N
    Y_expand = np.expand_dims(-(X - Mean), axis = 0) # 1 x m x N
    UMatrix = kernelMatrix * np.dot((X - Mean), (X - Mean).T) + np.sum(X_expand * gradKernel2, axis = -1) + np.sum(Y_expand * gradKernel1, axis = -1) + hessKernel

    return UMatrix

def True_KSD(mean, dim, bandwidth = True):
    if bandwidth == True:
        r = np.sqrt(dim)
    else:
        r = bandwidth
    d = dim
    u2 = np.dot(mean, mean)
    KSD = (r**d * u2) / (r**2 + 2)**(d/2)
    return KSD

def True_Variance(samplesize, mean, dim, bandwidth = True):
    if bandwidth == True:
        r = dim
    else:
        r = bandwidth
    d = dim
    u2 = np.dot(mean, mean)
    n = samplesize
    # l = math.log(r, dim)
    # s = math.log(u2, dim)
    # o = np.max([2-2*l, 1-l+s, 2*s])

    Cov_1 = (u2 * (r + 2) / (r + 1) + u2**2) * r**d / ((r + 1) * (r + 2))**(d / 2)
    Var_2 = (r / (r + 4))**(d / 2) * (d + 7 * d**2 / r**2 + 6 * d / r * u2 + u2**2 + 1 / d)
    return (4 * (n - 2) * Cov_1 + 2 * Var_2) / (n * (n - 1))


def Estimated_KSD(U):

    m, _ = U.shape
    matDiag = np.sum(U.diagonal())
    matSum = U.sum()
    KSD = (matSum - matDiag) / (m * (m - 1))
    
    return KSD

def Estimated_variance(U, mu):

    m, _ = U.shape
    MAT = (U - mu)**2
    matDiag = np.sum(MAT.diagonal())
    matSum = MAT.sum()
    V = (matSum - matDiag) / (m**2 * (m - 1)**2)
    
    return V

def Emp_variance(samplesize, dim, mean, iter = 200, set_bandwidth = True):
    
    m = samplesize
    var = np.identity(dim)
    V = np.zeros(iter)

    for i in tqdm(range(iter)):
        Multinormal_X = np.random.multivariate_normal(mean, var, m)
        UMatrix = UqMatrix(Multinormal_X, mean, set_bandwidth = set_bandwidth)
        mu = True_KSD(mean, dim, bandwidth = set_bandwidth)
        V[i] = Estimated_variance(UMatrix, mu)
    
    return np.mean(V)

def comparison_KSD(samplesize, dim, mean, iter = 200, set_bandwidth = True):

    var = np.identity(dim)
    S = np.zeros(iter)
    
    True_ksd = True_KSD(mean, dim, bandwidth = set_bandwidth)
    True_v = True_Variance(samplesize, mean, dim, bandwidth = set_bandwidth)
    Est_ksd = np.zeros(iter)
    Est_v = np.zeros(iter)
    for i in tqdm(range(iter)):
        Multinormal_X = np.random.multivariate_normal(mean, var, samplesize)
        UMatrix = UqMatrix(Multinormal_X, mean, set_bandwidth = set_bandwidth)
        Est_ksd[i] = Estimated_KSD(UMatrix)
        Est_v[i] = Estimated_variance(UMatrix, True_ksd)
    
    return True_ksd, True_v, Est_ksd, Est_v

def quantile_KSD(emp_KSD, alpha):
    n = len(emp_KSD)
    TFarray = emp_KSD[emp_KSD >= alpha]
    count = len(TFarray)
    return count / n




