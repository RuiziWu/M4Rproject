import numpy as np
from tqdm import tqdm
import KSD_test_functions as ksdF


def pValue_KLchange(samplesize, dim, KLstatus, meanvalue, bootstrapsize = 1000, iter = 200, MH_method = True, set_bandwidth = False):
    
    if KLstatus == "one const KL":
        mean = np.zeros(dim)
        mean[0] = meanvalue
    elif KLstatus == "all linear incre KL":
        mean = np.ones(dim) * meanvalue
    elif KLstatus == "one linear decre KL":
        mean = np.zeros(dim)
        mean[0] = meanvalue / np.sqrt(dim)
    elif KLstatus == "all linear decre KL":
        mean = np.ones(dim) * meanvalue / dim
    elif KLstatus == "one quadratic decre KL":
        mean = np.zeros(dim)
        mean[0] = meanvalue / dim
    elif KLstatus == "all nconst incre KL":
        mean = meanvalue / np.array([i for i in range(1, dim+1)])
    elif KLstatus == "one sqrt incre KL":
        mean = np.zeros(dim)
        mean[0] = meanvalue * (dim ** (1 / 4))
    elif KLstatus == "all sqrt incre KL":
        mean = np.ones(dim) * meanvalue / (dim ** (1 / 4))
    elif KLstatus == "one fourth incre KL":
        mean = np.zeros(dim)
        mean[0] = meanvalue * (dim ** (1 / 8))
    elif KLstatus == "one eighth incre KL":
        mean = np.zeros(dim)
        mean[0] = meanvalue * (dim ** (1 / 16))
    elif KLstatus == "one sixteenth incre KL":
        mean = np.zeros(dim)
        mean[0] = meanvalue * (dim ** (1 / 32))
    else:
        mean = np.zeros(dim)

    pvalue = np.zeros(iter)
    cov = np.identity(dim)
    
    for i in tqdm(range(iter)):
        Multinormal_X = np.random.multivariate_normal(mean, cov, samplesize)
        UMatrix = ksdF.UqMatrix(Multinormal_X, MH_method = MH_method, set_bandwidth = set_bandwidth)
        KSDvalue = ksdF.KSD(UMatrix)
        KSDstar = ksdF.Bootstrap_KSD(UMatrix, size = bootstrapsize)
        pvalue[i] = ksdF.approx_pvalue(KSDvalue, KSDstar)

    print("finish onemeanshift_constantKL")
    
    return pvalue