import numpy as np
from tqdm import tqdm
import KSD_test_functions as ksdF
import alternatives_Gaussian as h1g

samplesize = 1000
alpha = 0.05
BootstrapSize = 1000
iteration = 300
meanvalue = 0.1

dim_list = np.array([i*2 for i in range(11)])
dim_list[0] = 1
dim_list_len = len(dim_list)

# one eighth increase KL, mean of Xi = meanvalue / dim ** (1/4)
tp_KL = np.zeros(dim_list_len)
for i in tqdm(range(dim_list_len)):
    dim = dim_list[i]
    dim_p = h1g.pValue_KLchange(samplesize, dim, "one eighth incre KL", meanvalue, bootstrapsize = BootstrapSize, iter = iteration)
    tp_KL[i] = ksdF.test_power(dim_p, alpha)

np.savetxt("TP_oneeighth_iKL_iter300.csv", tp_KL, delimiter=",")

# one fourth increase KL, mean of Xi = meanvalue / dim ** (1/4)
tp_KL = np.zeros(dim_list_len)
for i in tqdm(range(dim_list_len)):
    dim = dim_list[i]
    dim_p = h1g.pValue_KLchange(samplesize, dim, "one sixteenth incre KL", meanvalue, bootstrapsize = BootstrapSize, iter = iteration)
    tp_KL[i] = ksdF.test_power(dim_p, alpha)

np.savetxt("TP_onesixteenth_iKL_iter300.csv", tp_KL, delimiter=",")
