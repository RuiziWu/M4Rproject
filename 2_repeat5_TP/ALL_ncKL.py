import numpy as np
import A1_functionKL2 as fKL

samplesize = 1000
alpha = 0.05
BootstrapSize = 1000
iteration = 100

dim_list = np.array([i*10 for i in range(11)])
dim_list[0] = 1
dim_list_len = len(dim_list)

# all mean of Xi shift, meanshift = 1 / i
tp_All_ncKL = np.zeros(dim_list_len)
for i in range(dim_list_len):
    dim = dim_list[i]
    dim_p = fKL.pValue_allmeanshift_notconstantKL(samplesize, dim, bootstrapsize= BootstrapSize, iter = 300)
    tp_All_ncKL[i] = fKL.test_power(dim_p, alpha)

np.savetxt("BTP_ALL_ncKL.csv", tp_All_ncKL, delimiter=",")