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

# all mean of Xi shift, meanshift = 0.01
tp_All_liKL = np.zeros((dim_list_len, repeat))
meanvalue = 0.01
for j in range(repeat):
    for i in range(dim_list_len):
        dim = dim_list[i]
        dim_p = fKL.pValue_allmeanshift_linearincreaseKL(samplesize, dim, meanvalue, bootstrapsize= BootstrapSize, iter = iteration)
        tp_All_liKL[i, j] = fKL.test_power(dim_p, alpha)
    print(f"repeat: {i + 1}")
np.savetxt("BTP_ALL_liKL.csv", tp_All_liKL, delimiter=",")