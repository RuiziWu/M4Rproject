import numpy as np
import A1_functionKL2 as fKL

samplesize = 1000
alpha = 0.05
BootstrapSize = 1000
iteration = 10

# dim_list = np.array([i*10 for i in range(11)])
dim_list = np.array([50])
# dim_list[0] = 1
dim_list_len = len(dim_list)

# all mean of Xi shift, meanshift = 1 / n
tp_All_ldKL = np.zeros(dim_list_len)

for i in range(dim_list_len):
    dim = dim_list[i]
    dim_p = fKL.pValue_allmeanshift_lineardecreaseKL(samplesize, dim, bootstrapsize = BootstrapSize, iter = iteration)
    tp_All_ldKL[i] = fKL.test_power(dim_p, alpha)

# np.savetxt("BTP_ALL_ldKL.csv", tp_All_ldKL, delimiter=",")


