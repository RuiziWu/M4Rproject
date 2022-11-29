import numpy as np
import M4RPROJECT.A1_funtionKL as fKL


samplesize = 1000
alpha = 0.05

dim_list = np.array([i*10 for i in range(1)])
dim_list[0] = 1
dim_list_len = len(dim_list)
for i in range(dim_list_len):
    dim = dim_list[i]
    dim_p = fKL.pValue_allmeanshift_notconstantKL(samplesize, dim, bootstrapsize = 1000, iter = 150)
    dim_tp = fKL.test_power(dim_p, alpha)

