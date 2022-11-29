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


def KSD(X, U):

    m, _ = X.shape
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
    for i in range(size):
        Weight = np.random.multinomial(m, multi_prob)
        Wadjust = (Weight - 1) / m
        WMatrix = np.outer(Wadjust, Wadjust)
        SMatrix = WMatrix * U
        diag_sum = sum(SMatrix.diagonal())
        matrix_sum = SMatrix.sum()
        Si = matrix_sum - diag_sum
        Sstar[i] = Si
        # if epochshow != False:
            # if (i+1) % epochshow == 0:
                # print(f"we are in epoch {i+1}")

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


def pValue_onemeanshift(samplesize, dim, meanvalue, bootstrapsize = 1000, iter = 100, MH_method = True, set_bandwidth = False):
    """
    param stepvalue: 1D numpy array with dimension dim or boolean value False
    param covalue: 1D numpy array with dimension dim or boolean value False

    param
    """
    n = len(meanvalue)
    pvalue = np.zeros((n, iter))
    cov = np.identity(dim)
    for i in range(n):
        mi = meanvalue[i]
        mean = np.zeros(dim)
        mean[0] = mi
        for j in range(iter):
            Multinormal_X = np.random.multivariate_normal(mean, cov, samplesize)
            UMatrix = UqMatrix(Multinormal_X, MH_method = MH_method, set_bandwidth = set_bandwidth)
            KSDvalue = KSD(Multinormal_X, UMatrix)
            KSDstar = Bootstrap_KSD(UMatrix, size = bootstrapsize, epochshow = False)
            pvalue[i, j] = approx_pvalue(KSDvalue, KSDstar)
        
    print("finish")
    
    return pvalue


def pValue_allmeanshift(samplesize, dim, meanvalue, bootstrapsize = 1000, iter = 100, MH_method = True, set_bandwidth = False):
    """
    param stepvalue: 1D numpy array with dimension dim or boolean value False
    param covalue: 1D numpy array with dimension dim or boolean value False

    param
    """
    n = len(meanvalue)
    pvalue = np.zeros((n, iter))
    cov = np.identity(dim)
    for i in range(n):
        mi = meanvalue[i]
        mean = np.repeat(mi, dim)
        for j in range(iter):
            Multinormal_X = np.random.multivariate_normal(mean, cov, samplesize)
            UMatrix = UqMatrix(Multinormal_X, MH_method = MH_method, set_bandwidth = set_bandwidth)
            KSDvalue = KSD(Multinormal_X, UMatrix)
            KSDstar = Bootstrap_KSD(UMatrix, size = bootstrapsize, epochshow = False)
            pvalue[i, j] = approx_pvalue(KSDvalue, KSDstar)
        
    print("finish!")
    
    return pvalue


def test_power(p, alpha):
    _, m = p.shape
    
    p2 = p.copy()
    # correctly rejects the null hypothesis
    p2[p < alpha] = 1
    # Type-II error
    p2[p >= alpha] = 0
    tp = np.sum(p2, axis = -1) / m
    return tp


meanshift = np.array([0.1])

# alpha = 0.05 (95% confidence interval)
alpha = 0.05
# 97.5% confidence interval
alpha2 = 0.025

dim1_om = pValue_onemeanshift(1000, 1, meanvalue = meanshift, iter = 300, MH_method = True)
tp_dim1_om = test_power(dim1_om, alpha)
tp_dim1_om2 = test_power(dim1_om, alpha2)
dim10_om = pValue_onemeanshift(1000, 10, meanvalue = meanshift, iter = 300, MH_method = True)
tp_dim10_om = test_power(dim10_om, alpha)
tp_dim10_om2 = test_power(dim10_om, alpha2)
dim20_om = pValue_onemeanshift(1000, 20, meanvalue = meanshift, iter = 300, MH_method = True)
tp_dim20_om = test_power(dim20_om, alpha)
tp_dim20_om2 = test_power(dim20_om, alpha2)
dim30_om = pValue_onemeanshift(1000, 30, meanvalue = meanshift, iter = 300, MH_method = True)
tp_dim30_om = test_power(dim30_om, alpha)
tp_dim30_om2 = test_power(dim30_om, alpha2)
dim40_om = pValue_onemeanshift(1000, 40, meanvalue = meanshift, iter = 300, MH_method = True)
tp_dim40_om = test_power(dim40_om, alpha)
tp_dim40_om2 = test_power(dim40_om, alpha2)
dim50_om = pValue_onemeanshift(1000, 50, meanvalue = meanshift, iter = 300, MH_method = True)
tp_dim50_om = test_power(dim50_om, alpha)
tp_dim50_om2 = test_power(dim50_om, alpha2)
dim60_om = pValue_onemeanshift(1000, 60, meanvalue = meanshift, iter = 300, MH_method = True)
tp_dim60_om = test_power(dim60_om, alpha)
tp_dim60_om2 = test_power(dim60_om, alpha2)
dim70_om = pValue_onemeanshift(1000, 70, meanvalue = meanshift, iter = 300, MH_method = True)
tp_dim70_om = test_power(dim70_om, alpha)
tp_dim70_om2 = test_power(dim70_om, alpha2)
dim80_om = pValue_onemeanshift(1000, 80, meanvalue = meanshift, iter = 300, MH_method = True)
tp_dim80_om = test_power(dim80_om, alpha)
tp_dim80_om2 = test_power(dim80_om, alpha2)
dim90_om = pValue_onemeanshift(1000, 90, meanvalue = meanshift, iter = 300, MH_method = True)
tp_dim90_om = test_power(dim90_om, alpha)
tp_dim90_om2 = test_power(dim90_om, alpha2)
dim100_om = pValue_onemeanshift(1000, 100, meanvalue = meanshift, iter = 300, MH_method = True)
tp_dim100_om = test_power(dim100_om, alpha)
tp_dim100_om2 = test_power(dim100_om, alpha2)
tp_om = np.array([tp_dim1_om, tp_dim10_om, tp_dim20_om, tp_dim30_om, tp_dim40_om, tp_dim50_om, tp_dim60_om, tp_dim70_om, tp_dim80_om, tp_dim90_om, tp_dim100_om])
tp_om2 = np.array([tp_dim1_om2, tp_dim10_om2, tp_dim20_om2, tp_dim30_om2, tp_dim40_om2, tp_dim50_om2, tp_dim60_om2, tp_dim70_om2, tp_dim80_om2, tp_dim90_om2, tp_dim100_om2])
p_om = np.array([dim1_om[0], dim10_om[0], dim20_om[0], dim30_om[0], dim40_om[0], dim50_om[0], dim60_om[0], dim70_om[0], dim80_om[0], dim90_om[0], dim100_om[0]])
np.savetxt("TP_ONE_MS.csv", tp_om, delimiter=",")
np.savetxt("TP_ONE_MS2.csv", tp_om2, delimiter=",")
np.savetxt("TP_ONE_P.csv", p_om, delimiter=",")

dim1_am = pValue_allmeanshift(1000, 1, meanvalue = meanshift, iter = 300, MH_method = True)
tp_dim1_am = test_power(dim1_am, alpha)
tp_dim1_am2 = test_power(dim1_am, alpha2)
dim10_am = pValue_allmeanshift(1000, 10, meanvalue = meanshift, iter = 300, MH_method = True)
tp_dim10_am = test_power(dim10_am, alpha)
tp_dim10_am2 = test_power(dim10_am, alpha2)
dim20_am = pValue_allmeanshift(1000, 20, meanvalue = meanshift, iter = 300, MH_method = True)
tp_dim20_am = test_power(dim20_am, alpha)
tp_dim20_am2 = test_power(dim20_am, alpha2)
dim30_am = pValue_allmeanshift(1000, 30, meanvalue = meanshift, iter = 300, MH_method = True)
tp_dim30_am = test_power(dim30_am, alpha)
tp_dim30_am2 = test_power(dim30_am, alpha2)
dim40_am = pValue_allmeanshift(1000, 40, meanvalue = meanshift, iter = 300, MH_method = True)
tp_dim40_am = test_power(dim40_am, alpha)
tp_dim40_am2 = test_power(dim40_am, alpha2)
dim50_am = pValue_allmeanshift(1000, 50, meanvalue = meanshift, iter = 300, MH_method = True)
tp_dim50_am = test_power(dim50_am, alpha)
tp_dim50_am2 = test_power(dim50_am, alpha2)
dim60_am = pValue_allmeanshift(1000, 60, meanvalue = meanshift, iter = 300, MH_method = True)
tp_dim60_am = test_power(dim60_am, alpha)
tp_dim60_am2 = test_power(dim60_am, alpha2)
dim70_am = pValue_allmeanshift(1000, 70, meanvalue = meanshift, iter = 300, MH_method = True)
tp_dim70_am = test_power(dim70_am, alpha)
tp_dim70_am2 = test_power(dim70_am, alpha2)
dim80_am = pValue_allmeanshift(1000, 80, meanvalue = meanshift, iter = 300, MH_method = True)
tp_dim80_am = test_power(dim80_am, alpha)
tp_dim80_am2 = test_power(dim80_am, alpha2)
dim90_am = pValue_allmeanshift(1000, 90, meanvalue = meanshift, iter = 300, MH_method = True)
tp_dim90_am = test_power(dim90_am, alpha)
tp_dim90_am2 = test_power(dim90_am, alpha2)
dim100_am = pValue_allmeanshift(1000, 100, meanvalue = meanshift, iter = 300, MH_method = True)
tp_dim100_am = test_power(dim100_am, alpha)
tp_dim100_am2 = test_power(dim100_am, alpha2)
tp_am = np.array([tp_dim1_am, tp_dim10_am, tp_dim20_am, tp_dim30_am, tp_dim40_am, tp_dim50_am, tp_dim60_am, tp_dim70_am, tp_dim80_am, tp_dim90_am, tp_dim100_am])
tp_am2 = np.array([tp_dim1_am2, tp_dim10_am2, tp_dim20_am2, tp_dim30_am2, tp_dim40_am2, tp_dim50_am2, tp_dim60_am2, tp_dim70_am2, tp_dim80_am2, tp_dim90_am2, tp_dim100_am2])
p_am = np.array([dim1_am[0], dim10_am[0], dim20_am[0], dim30_am[0], dim40_am[0], dim50_am[0], dim60_am[0], dim70_am[0], dim80_am[0], dim90_am[0], dim100_am[0]])
np.savetxt("TP_ALL_MS.csv", tp_am, delimiter=",")
np.savetxt("TP_ALL_MS2.csv", tp_am2, delimiter=",")
np.savetxt("TP_ALL_P.csv", p_am, delimiter=",")
