import numpy as np
import matplotlib.pyplot as plt

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
        if epochshow != False:
            if (i+1) % epochshow == 0:
                print(f"we are in epoch {i+1}")

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

def pValue_meanshift(samplesize, dim, bandwidth, meanvalue, bootstrapsize = 1000, iter = 100):
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
            UMatrix = UqMatrix(Multinormal_X, bandwidth)
            KSDvalue = KSD(Multinormal_X, UMatrix)
            KSDstar = Bootstrap_KSD(UMatrix, size = bootstrapsize, epochshow = False)
            pvalue[i, j] = approx_pvalue(KSDvalue, KSDstar)
        
        print(f"the {i + 1}th mean finished !")
    return pvalue

zeromean = np.array([0])

dim40_sieze100 = pValue_meanshift(100, 40, 1, meanvalue=zeromean, bootstrapsize = 5000, iter = 300)
dim40_sieze500 = pValue_meanshift(500, 40, 1, meanvalue=zeromean, bootstrapsize = 5000, iter = 300)
dim40_sieze1000 = pValue_meanshift(1000, 40, 1, meanvalue=zeromean, bootstrapsize = 5000, iter = 300)
dim40_sieze1500 = pValue_meanshift(1500, 40, 1, meanvalue=zeromean, bootstrapsize = 5000, iter = 300)
dim40_sieze2000 = pValue_meanshift(2000, 20, 1, meanvalue=zeromean, bootstrapsize = 5000, iter = 300)

data = np.array([dim40_sieze100[0], dim40_sieze500[0], dim40_sieze1000[0], dim40_sieze1500[0], dim40_sieze2000[0]])

fig = plt.figure(figsize =(12, 6))
fig.suptitle("p value agains change of sample size", fontsize = 16)
ax = fig.add_subplot(111)
ax.boxplot(data.T)
ax.set_xticklabels(["100", "500", "1000", "1500", "2000"])
ax.set_xlabel("sample size")
ax.set_ylabel("p value")
plt.show()
