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


def kernel_function(X, set_bandwidth = True):
    """
    Args:
        X: m x N
        h: float
    """

    _, N = X.shape

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


def Matrix_Up(X, set_bandwidth = True):
    
    kernelMatrix, gradKernel1, gradKernel2, hessKernel = kernel_function(X, set_bandwidth = set_bandwidth)
    X_expand = np.expand_dims(-X, axis = 1) # m x 1 x N
    Y_expand = np.expand_dims(-X, axis = 0) # 1 x m x N
    UpMatrix = kernelMatrix * np.dot(X, X.T) + np.sum(X_expand * gradKernel2, axis = -1) + np.sum(Y_expand * gradKernel1, axis = -1) + hessKernel

    return UpMatrix


def True_KSD(mean, dim, bandwidth = True):
    if bandwidth == True:
        r = np.sqrt(dim / 2)
    else:
        r = bandwidth
    d = dim
    u2 = np.dot(mean, mean)
    KSD = (r**2 / (r**2 + 2))**(d/2) * u2
    return KSD


def Estimated_KSD(U):
    """
    Unbiased estimator of KSD
    """
    m, _ = U.shape # m = samplesize
    matDiag = np.sum(U.diagonal())
    matSum = U.sum()
    KSD = (matSum - matDiag) / (m * (m - 1))
    
    return KSD

def Emp_Expectation(samplesize, dim, mean, bandwidth=True):
    
    var = np.identity(dim)
    Multinormal_X = np.random.multivariate_normal(mean, var, samplesize)
    UpMatrix = Matrix_Up(Multinormal_X, set_bandwidth = bandwidth)
    Est_ksd = Estimated_KSD(UpMatrix)

    return Est_ksd


def True_Variance(samplesize, dim, mean, bandwidth = True):
    if bandwidth == True:
        r = np.sqrt(dim / 2) ** 2
    else:
        r = bandwidth ** 2
    m = samplesize
    d = dim
    u2 = np.dot(mean, mean)

    sigma_cond = (r**2 / ((1 + r) * (3 + r))) ** (d / 2) * ((2 + r)**2 / ((1 + r) * (3 + r)) * u2 + (1 - ((1 + r) * (3 + r) / (2 + r)**2)**(d/ 2)) * u2**2)
    sigma_full = (r / (4 + r))**(d / 2) * (d + d**2 / r + 2 * d * u2 / r + 2 * u2 + (1 - (r * (4 + r) / (2 + r)**2)**(d / 2)) * u2**2)

    return (4 * (m - 2) * sigma_cond + 2 * sigma_full) / (m * (m - 1))


def Emp_Variance(samplesize, dim, mean, iteration=300, bandwidth=True):
    
    var = np.identity(dim)
    Est_ksd = np.zeros(iteration)

    for i in tqdm(range(iteration)):
        Multinormal_X = np.random.multivariate_normal(mean, var, samplesize)
        UpMatrix = Matrix_Up(Multinormal_X, set_bandwidth = bandwidth)
        Est_ksd[i] = Estimated_KSD(UpMatrix)

    var = Est_ksd.var()
    print(var)
    return var


def Unbiased_KSDs(samplesize, dim, mean, iteration=300, bandwidth=True):

    var = np.identity(dim)
    Est_ksd = np.zeros(iteration)

    for i in tqdm(range(iteration)):
        Multinormal_X = np.random.multivariate_normal(mean, var, samplesize)
        UpMatrix = Matrix_Up(Multinormal_X, set_bandwidth = bandwidth)
        Est_ksd[i] = Estimated_KSD(UpMatrix)

    return Est_ksd


def quantile_KSD(emp_KSD, alpha):

    n = len(emp_KSD)
    TFarray = emp_KSD[emp_KSD >= alpha]
    count = len(TFarray)
    
    return count / n


# def Verify_Variance(samplesize, dim, mean, bandwidth=True):
#     if bandwidth == True:
#         h = np.sqrt(dim / 2)
#     else:
#         h = bandwidth
#     d = dim
#     u2 = np.dot(mean, mean)

#     coeff_d2 = (9 * h**8 + 24 * h**6 + 32 * h**4 + 28 * h**2 + 24) / (h**8 * (h**2 + 4)**2)
#     coeff_d = (h**12 - 8 * h**10 - 116 * h**8 - 432 * h**6 - 736 * h**4 - 640 * h**2 - 224) / (h**8 * (h**2 + 4)**2)
#     coeff_du2 = 2 * (h**10 - 22 * h**6 - 56 * h**4 - 50 * h**2 - 16) / (h**8 * (h**2 + 2) * (h**2 + 4))
#     coeff_u2 = 2 * (h**12 + 6 * h**10 - 52 * h**6 - 112 * h**4 - 100 * h**2 -32) / (h**8 * (h**2 + 2) * (h**2 + 4))
#     coeff_u4 = (h**12 + 4 * h**10 - 28 * 10**6 - 60 * h**4 - 52 * h**2 - 16) / (h**8 * (h**2 + 2)**2)

#     true_var = coeff_d2 * d**2 + coeff_d * d + coeff_du2 * d * u2 + coeff_u2 * u2 + coeff_u4 * u2**2

#     var = np.identity(dim)
#     Multinormal_X = np.random.multivariate_normal(mean, var, samplesize)
#     UpMatrix = Matrix_Up(Multinormal_X, set_bandwidth = bandwidth)

#     return UpMatrix.var(), true_var








