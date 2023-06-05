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


def kernel_function(X, set_bandwidth):
    """
    Args:
        X: m x N
        h: float
    """

    _, N = X.shape

    diff_norm_sq = diff_norm_sq_fn(X, X) # m x m
    if set_bandwidth == 'Med':
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


def Matrix_Up(X, set_bandwidth):
    
    kernelMatrix, gradKernel1, gradKernel2, hessKernel = kernel_function(X, set_bandwidth)
    X_expand = np.expand_dims(-X, axis = 1) # m x 1 x N
    Y_expand = np.expand_dims(-X, axis = 0) # 1 x m x N
    UpMatrix = kernelMatrix * np.dot(X, X.T) + np.sum(X_expand * gradKernel2, axis = -1) + np.sum(Y_expand * gradKernel1, axis = -1) + hessKernel

    return UpMatrix


def True_KSD(mean, dim, bandwidth):
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

def Emp_Expectation(samplesize, dim, mean, bandwidth):
    
    var = np.identity(dim)
    Multinormal_X = np.random.multivariate_normal(mean, var, samplesize)
    UpMatrix = Matrix_Up(Multinormal_X, set_bandwidth = bandwidth)
    Est_ksd = Estimated_KSD(UpMatrix)

    return Est_ksd


def True_Variance(samplesize, dim, mean, bandwidth):
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


def Emp_Variance(samplesize, dim, mean, bandwidth, iteration=300):
    
    var = np.identity(dim)
    Est_ksd = np.zeros(iteration)

    for i in tqdm(range(iteration)):
        Multinormal_X = np.random.multivariate_normal(mean, var, samplesize)
        UpMatrix = Matrix_Up(Multinormal_X, set_bandwidth = bandwidth)
        Est_ksd[i] = Estimated_KSD(UpMatrix)

    var = Est_ksd.var()
    print(var)
    return var


def Unbiased_KSDs(samplesize, dim, mean, bandwidth, iteration=300):

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


def Bootstrap_KSD(U, size = 1000):

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


def approx_pvalue(S, Sstar): # rejection rate
    """
    param S: unbiased estimation of KSD, scalar
    param Sstar: unbiased m bootstrap sample KSD
    """
    n = len(Sstar)
    TFarray = Sstar[Sstar >= S]
    count = len(TFarray)
    return count / n


def pDist_H0_dimchange(samplesize, dim_list, set_bandwidth, bootstrapsize = 1000, iter = 100):
    """
    dim: a list of dimensions
    """
    n = len(dim_list)
    pvalue = np.zeros((n, iter))
    
    for i in tqdm(range(n)):
        dim = dim_list[i]
        mean = np.zeros(dim)
        cov = np.identity(dim)
        for j in tqdm(range(iter)):
            Multinormal_X = np.random.multivariate_normal(mean, cov, samplesize)
            UMatrix = Matrix_Up(Multinormal_X, set_bandwidth)
            KSDvalue = Estimated_KSD(UMatrix)
            KSDstar = Bootstrap_KSD(UMatrix, size = bootstrapsize)
            pvalue[i, j] = approx_pvalue(KSDvalue, KSDstar)

    return pvalue



def pDist_H1_dimchange(samplesize, dim_list, meanshift, set_bandwidth, bootstrapsize = 1000, iter = 100):
    """
    dim: a list of dimensions
    """
    n = len(dim_list)
    pvalue = np.zeros((n, iter))
    
    for i in tqdm(range(n)):
        dim = dim_list[i]
        mean = np.zeros(dim)
        mean[0] = meanshift
        cov = np.identity(dim)
        for j in tqdm(range(iter)):
            Multinormal_X = np.random.multivariate_normal(mean, cov, samplesize)
            assert Multinormal_X.shape == (samplesize, dim)
            UMatrix = Matrix_Up(Multinormal_X, set_bandwidth = set_bandwidth)
            KSDvalue = Estimated_KSD(UMatrix)
            KSDstar = Bootstrap_KSD(UMatrix, size = bootstrapsize)
            pvalue[i, j] = approx_pvalue(KSDvalue, KSDstar)

    return pvalue

