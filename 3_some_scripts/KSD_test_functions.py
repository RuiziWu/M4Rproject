import numpy as np


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
    # Weight_all = np.random.multinomial(m, multi_prob, size=size)
    # Weight_adj = (Weight_all - 1) / m

    for i in range(size):
    #     Weight = Weight_adj[i]
    #     WMatrix = np.outer(Weight, Weight)
    #     SMatrix = WMatrix * U
    #     diag_sum = sum(SMatrix.diagonal())
    #     matrix_sum = SMatrix.sum()
    #     Si = matrix_sum - diag_sum
    #     Sstar[i] = Si
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


def test_power(p, alpha):
    
    tp = np.mean(p <= alpha, axis = -1)
    
    return tp