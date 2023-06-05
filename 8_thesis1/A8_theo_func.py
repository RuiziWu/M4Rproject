import numpy as np
import A8_functions as A8
from tqdm import tqdm

def func_C0(d):
    C0 = (d / (d + 4))**(d/2)
    return C0


def func_C1(d):
    C1 = 2 * (d / (d+8))**(d/2) * ((d**4 + 32 * d**3 + 116 * d**2 + 228 * d + 16) / (d * (d + 8)**2))
    return C1


def func_C2(d):
    C2 = 2 * (d / (d+8))**(d/2) * (2 + 8 * d / (d + 8))
    return C2


def func_C3(d):
    C3 = 2 * (d / (d+8))**(d/2) * (1 - ((d * (d + 8)) / (4 * (d + 4)**2))**(d/2))
    return C3


def func_C4(d):
    C4 = (d**2 / ((d + 2) * (d + 6)))**(d/2) * 8 * (d + 4)**2 / ((d + 2) * (d + 6))
    return C4


def func_C5(d):
    C5 = (d**2 / ((d + 2) * (d + 6)))**(d/2) * (1 - (((d + 2) * (d + 6)) / (d + 4)**2)**(d/2))
    return C5


def func_C6(d):
    C2 = func_C2(d)
    C4 = func_C4(d)
    C6 = np.sqrt(C2 + 4 * C4)
    return C6


def func_C7(d):
    C3 = func_C3(d)
    C5 = func_C5(d)
    C7 = np.sqrt(C3 + 4 * C5)
    return C7


def func_U1(Delta2):
    C2 = func_C2(2)
    C3 = func_C3(2)
    U1 = C2 * Delta2 + C3 * Delta2**2
    return U1


def func_U2(Delta2):
    C4 = func_C4(2)
    C5 = func_C5(2)
    U2 = 4 * (C4 * Delta2 + C5 * Delta2**2)
    return U2


def func_U3(Delta2, beta):
    Delta = np.sqrt(Delta2)
    U2 = func_U2(Delta2)
    U3 = (np.sqrt(np.exp(2) * beta**(-1) * Delta**(-1) * U2) - Delta * np.exp(-1)) / 2
    return U3


def LowerBound_nvsdim(dim, mean, alpha, beta):

    d = dim
    Delta2 = np.dot(mean, mean)

    C1 = func_C1(d)
    U1 = func_U1(Delta2)
    U3 = func_U3(Delta2, beta)
    term1 = np.sqrt(2 * C1 * (alpha + beta) / (alpha * beta))
    term2 = np.sqrt(2 * U1 / beta)

    LB_n = np.exp(2) * Delta2**(-1) * (( term1 + term2 + U3**2 )**(1/2) + U3)**2 + 1

    return int(np.ceil(LB_n))


def LowerBound_nvsmean(dim, mean, alpha, beta):
    d = dim
    Delta2 = np.dot(mean, mean)

    C0 = func_C0(d)
    C1 = func_C1(d)
    C6 = func_C6(d)
    C7 = func_C7(d)
    C6C7 = C6 * np.sqrt(Delta2) + C7 * Delta2
    C = np.sqrt(2 * C1 / alpha) + np.sqrt(2 * C1 / beta)
    
    term1 = C0 * Delta2 / C
    term2 = C6C7**2 / (4 * beta * C**2)
    term3 = C6C7 / (2 * np.sqrt(beta) * C)

    LB_n = ((term1 + term2)**(1/2) - term3)**(-2) + 1

    return int(np.ceil(LB_n))


def pDist_nvsdim(dim_list, meanshift, alpha, beta, bootstrapsize = 1000, iter = 100):
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
        samplesize = LowerBound_nvsdim(dim, mean, alpha, beta)
        for j in tqdm(range(iter)):
            Multinormal_X = np.random.multivariate_normal(mean, cov, samplesize)
            UMatrix = A8.Matrix_Up(Multinormal_X, 'Med')
            KSDvalue = A8.Estimated_KSD(UMatrix)
            KSDstar = A8.Bootstrap_KSD(UMatrix, size = bootstrapsize)
            pvalue[i, j] = A8.approx_pvalue(KSDvalue, KSDstar)

    return pvalue


def pDist_nvsmean(dim, mean_list, alpha, beta, bootstrapsize = 1000, iter = 100):
    """
    dim: a list of dimensions
    """
    n = len(mean_list)
    pvalue = np.zeros((n, iter))
    
    for i in tqdm(range(n)):
        mean = np.zeros(dim)
        mean[0] = mean_list[i]
        cov = np.identity(dim)
        samplesize = LowerBound_nvsmean(dim, mean, alpha, beta)
        for j in tqdm(range(iter)):
            Multinormal_X = np.random.multivariate_normal(mean, cov, samplesize)
            UMatrix = A8.Matrix_Up(Multinormal_X, 'Med')
            KSDvalue = A8.Estimated_KSD(UMatrix)
            KSDstar = A8.Bootstrap_KSD(UMatrix, size = bootstrapsize)
            pvalue[i, j] = A8.approx_pvalue(KSDvalue, KSDstar)

    return pvalue




