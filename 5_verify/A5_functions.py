import numpy as np
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


def kernel_function(X, MH_method = True, set_bandwidth = False):
    """
    Args:
        X: m x N
        h: float
    """

    m, N = X.shape

    diff_norm_sq = diff_norm_sq_fn(X, X) # m x m
    if MH_method == True:
        h = np.round(np.median(diff_norm_sq[np.triu_indices(m, k=1)]), 6)

    else:
        h = set_bandwidth
        
    kernelMatrix = np.exp(- 1 / (2 * h**2) * diff_norm_sq) # m x m

    kernelMatrix_expand  = np.expand_dims(kernelMatrix, axis=-1) # m x m x 1
    diff = diff_fn(X, X) # m x m x N
    gradKernel1 = - 1 / (h**2) * diff * kernelMatrix_expand # m x m x N
    gradKernel2 = - gradKernel1 # m x m x N

    hessKernel = (N - (1 / h **2) * diff_norm_sq) * kernelMatrix / h ** 2
    
    return kernelMatrix, gradKernel1, gradKernel2, hessKernel


def UqMatrix(X, MH_method = True, set_bandwidth = False):
    
    kernelMatrix, gradKernel1, gradKernel2, hessKernel = kernel_function(X, MH_method = MH_method, set_bandwidth = set_bandwidth)
    X_expand = np.expand_dims(-X, axis = 1) # m x 1 x N
    Y_expand = np.expand_dims(-X, axis = 0) # 1 x m x N
    UMatrix = kernelMatrix * np.dot(X, X.T) + np.sum(X_expand * gradKernel2, axis = -1) + np.sum(Y_expand * gradKernel1, axis = -1) + hessKernel

    return UMatrix


def Unbiased_KSD(U):

    m, _ = U.shape
    matDiag = np.sum(U.diagonal())
    matSum = U.sum()
    KSD = (matSum - matDiag) / (m * (m - 1))
    
    return KSD

def E_KSD(samplesize, dim, mean, iter = 200, MH_method = False, set_bandwidth = 0.1):

    var = np.identity(dim)
    S = np.zeros(iter)

    for i in tqdm(range(iter)):
        Multinormal_X = np.random.multivariate_normal(mean, var, samplesize)
        UMatrix = UqMatrix(Multinormal_X, MH_method = MH_method, set_bandwidth = set_bandwidth)
        KSDvalue = Unbiased_KSD(UMatrix)
        S[i] = KSDvalue
    
    return np.mean(S)

def KSD_values(samplesize, dim, mean, iter = 200, MH_method = True, set_bandwidth = False):

    var = np.identity(dim)
    S = np.zeros(iter)

    for i in tqdm(range(iter)):
        Multinormal_X = np.random.multivariate_normal(mean, var, samplesize)
        UMatrix = UqMatrix(Multinormal_X, MH_method = MH_method, set_bandwidth = set_bandwidth)
        KSDvalue = Unbiased_KSD(UMatrix)
        S[i] = KSDvalue
    
    return UMatrix, S

def quantile_KSD(emp_KSD, alpha):
    n = len(emp_KSD)
    TFarray = emp_KSD[emp_KSD >= alpha]
    count = len(TFarray)
    return count / n

def True_KSD(mean, dim):
    r = np.sqrt(dim) / 2
    KSD = (r**dim * np.dot(mean, mean)) / (r**2 + 2)**(dim/2)
    return KSD

def Var_KSD(samplesize, mean, dim):
    n = samplesize
    u = np.dot(mean, mean)
    d = dim
    r = (np.sqrt(dim) / 2)**2
    Cov_1 = (u * (r + 2) / (r + 1) + u**2) * r**d / ((r + 1) * (r + 2))**(d / 2)
    Var_2 = (d + u**2 + d) * r**(d / 2) / (r + 4)**(d / 2)
    return (4 * (n - 2) * Cov_1 + 2 * Var_2) / (n * (n - 1))




