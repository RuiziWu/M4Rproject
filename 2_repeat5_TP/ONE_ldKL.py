import numpy as np
import A1_functionKL2 as fKL

samplesize = 1000
alpha = 0.05
BootstrapSize = 1000
iteration = 200
repeat = 5

dim_list = np.array([i*10 for i in range(11)])
dim_list[0] = 1
dim_list_len = len(dim_list)

# only mean of X1 shift, meanshift = 1 / n^0.5
tp_One_ldKL = np.zeros((dim_list_len, repeat))
for j in range(repeat):
    for i in range(dim_list_len):
        dim = dim_list[i]
        dim_p = fKL.pValue_onemeanshift_lineardecreaseKL(samplesize, dim, bootstrapsize= BootstrapSize, iter = iteration)
        tp_One_ldKL[i, j] = fKL.test_power(dim_p, alpha)
    print(f"repeat: {i + 1}")
np.savetxt("BTP_ONE_ldKL.csv", tp_One_ldKL, delimiter=",")