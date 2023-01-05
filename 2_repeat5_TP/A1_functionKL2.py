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


def KSD(X, U):

    m, _ = U.shape
    matDiag = np.sum(U.diagonal())
    matSum = U.sum()
    KSD = (matSum - matDiag) / (m * (m - 1))
    
    return KSD


def Bootstrap_KSD(U, size = 1000, epochshow = False):
    """
    
    """

    m, _ = U.shape
    multi_prob = np.repeat((1 / m), m)

    Sstar = np.zeros(size)
    Weight_all = np.random.multinomial(m, multi_prob, size=size)
    Weight_adj = (Weight_all - 1) / m

    for i in range(size):
        Weight = Weight_adj[i]
        WMatrix = np.outer(Weight, Weight)
        SMatrix = WMatrix * U
        diag_sum = sum(SMatrix.diagonal())
        matrix_sum = SMatrix.sum()
        Si = matrix_sum - diag_sum
        Sstar[i] = Si

    return Sstar


# def Bootstrap_KSD(U, size = 1000, epochshow = False):
#     """
    
#     """

#     m, _ = U.shape
#     multi_prob = np.repeat((1 / m), m)

#     Sstar = np.zeros(size)
#     Weight_all = np.random.multinomial(m, multi_prob, size=size)
#     Weight_adj = (Weight_all - 1) / m
#     Weight_x = np.expand_dims(Weight_adj, axis=1)
#     Weight_y = np.expand_dims(Weight_adj, axis=-1)
#     Weight = Weight_x * Weight_y
#     U_new = np.expand_dims(U, axis=0)
#     SMatrix = Weight * U_new
#     diag_sum = np.sum(np.diagonal(SMatrix, axis1=1, axis2=-1), axis=-1)
#     matrix_sum = np.sum(SMatrix, axis=(1, 2))
#     Sstar = matrix_sum - diag_sum

#     return Sstar


def approx_pvalue(S, Sstar): # rejection rate
    """
    param S: unbiased estimation of KSD, scalar
    param Sstar: unbiased m bootstrap sample KSD
    """
    n = len(Sstar)
    TFarray = Sstar[Sstar >= S]
    count = len(TFarray)
    return count / n


def test_power(p, alpha):
    # m = len(p)
    
    # p2 = p.copy()
    # # correctly rejects the null hypothesis
    # p2[p < alpha] = 1
    # # Type-II error
    # p2[p >= alpha] = 0
    # tp = np.sum(p2, axis = -1) / m
    tp = np.mean(p <= alpha, axis = -1)
    
    return tp


def pValue_onemeanshift_constantKL(samplesize, dim, meanvalue, bootstrapsize = 1000, iter = 100, MH_method = True, set_bandwidth = False):
    """

    param
    """
    
    pvalue = np.zeros(iter)
    cov = np.identity(dim)
    mean = np.zeros(dim)
    mean[0] = meanvalue
    for i in range(iter):
        Multinormal_X = np.random.multivariate_normal(mean, cov, samplesize)
        UMatrix = UqMatrix(Multinormal_X, MH_method = MH_method, set_bandwidth = set_bandwidth)
        KSDvalue = KSD(Multinormal_X, UMatrix)
        KSDstar = Bootstrap_KSD(UMatrix, size = bootstrapsize, epochshow = False)
        pvalue[i] = approx_pvalue(KSDvalue, KSDstar)

    print("finish onemeanshift_constantKL")
    
    return pvalue


def pValue_allmeanshift_linearincreaseKL(samplesize, dim, meanvalue, bootstrapsize = 1000, iter = 100, MH_method = True, set_bandwidth = False):
    """

    param
    """

    pvalue = np.zeros(iter)
    cov = np.identity(dim)
    mean = np.repeat(meanvalue, dim)
    for i in range(iter):
        Multinormal_X = np.random.multivariate_normal(mean, cov, samplesize)
        UMatrix = UqMatrix(Multinormal_X, MH_method = MH_method, set_bandwidth = set_bandwidth)
        KSDvalue = KSD(Multinormal_X, UMatrix)
        KSDstar = Bootstrap_KSD(UMatrix, size = bootstrapsize, epochshow = False)
        pvalue[i] = approx_pvalue(KSDvalue, KSDstar)
        
    print("finish! allmeanshift_linearincreaseKL")
    
    return pvalue


def pValue_onemeanshift_lineardecreaseKL(samplesize, dim, bootstrapsize = 1000, iter = 100, MH_method = True, set_bandwidth = False):

    pvalue = np.zeros(iter)
    cov = np.identity(dim)
    mean = np.zeros(dim)
    mean[0] = 1 / np.sqrt(dim)
    for i in range(iter):
        Multinormal_X = np.random.multivariate_normal(mean, cov, samplesize)
        UMatrix = UqMatrix(Multinormal_X, MH_method = MH_method, set_bandwidth = set_bandwidth)
        KSDvalue = KSD(Multinormal_X, UMatrix)
        KSDstar = Bootstrap_KSD(UMatrix, size = bootstrapsize, epochshow = False)
        pvalue[i] = approx_pvalue(KSDvalue, KSDstar)
        
    print("finish! onemeanshift_lineardecreaseKL")
    
    return pvalue


def pValue_allmeanshift_lineardecreaseKL(samplesize, dim, bootstrapsize = 1000, iter = 100, MH_method = True, set_bandwidth = False):

    pvalue = np.zeros(iter)
    cov = np.identity(dim)
    mean = np.ones(dim) / dim
    Multinormal_Xall = np.random.multivariate_normal(mean, cov, (iter, samplesize))
    for i in range(iter):
        # Multinormal_X = np.random.multivariate_normal(mean, cov, samplesize)
        Multinormal_X = Multinormal_Xall[i]
        UMatrix = UqMatrix(Multinormal_X, MH_method = MH_method, set_bandwidth = set_bandwidth)
        KSDvalue = KSD(Multinormal_X, UMatrix)
        KSDstar = Bootstrap_KSD(UMatrix, size = bootstrapsize, epochshow = False)
        pvalue[i] = approx_pvalue(KSDvalue, KSDstar)
        
    print("finish! allmeanshift_lineardecreaseKL")
    
    return pvalue


def pValue_onemeanshift_quaddecreaseKL(samplesize, dim, bootstrapsize = 1000, iter = 100, MH_method = True, set_bandwidth = False):

    pvalue = np.zeros(iter)
    cov = np.identity(dim)
    mean = np.zeros(dim)
    mean[0] = 1 / dim
    for i in range(iter):
        Multinormal_X = np.random.multivariate_normal(mean, cov, samplesize)
        UMatrix = UqMatrix(Multinormal_X, MH_method = MH_method, set_bandwidth = set_bandwidth)
        KSDvalue = KSD(Multinormal_X, UMatrix)
        KSDstar = Bootstrap_KSD(UMatrix, size = bootstrapsize, epochshow = False)
        pvalue[i] = approx_pvalue(KSDvalue, KSDstar)
        
    print("finish! onemeanshift_quaddecreaseKL")
    
    return pvalue


def pValue_allmeanshift_notconstantKL(samplesize, dim, bootstrapsize = 1000, iter = 100, MH_method = True, set_bandwidth = False):

    pvalue = np.zeros(iter)
    cov = np.identity(dim)
    k_array = np.array([i for i in range(1, dim+1)])
    mean = np.ones(dim) * 0.01 / k_array
    for i in range(iter):
        Multinormal_X = np.random.multivariate_normal(mean, cov, samplesize)
        UMatrix = UqMatrix(Multinormal_X, MH_method = MH_method, set_bandwidth = set_bandwidth)
        KSDvalue = KSD(Multinormal_X, UMatrix)
        KSDstar = Bootstrap_KSD(UMatrix, size = bootstrapsize, epochshow = False)
        pvalue[i] = approx_pvalue(KSDvalue, KSDstar)
        
    print("finish! allmeanshift_notconstantKL")
    
    return pvalue