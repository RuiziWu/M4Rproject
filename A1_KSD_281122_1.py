import numpy as np
import A1_functionKL as fKL

samplesize = 1000
alpha = 0.05
BootstrapSize = 1000
iteration = 150
repeat = 5

dim_list = np.array([i*10 for i in range(11)])
dim_list[0] = 1
dim_list_len = len(dim_list)



# only mean of X1 shift, meanshift = 1 / n^0.5
# p_One_ldKL = np.zeros((dim_list_len, iteration))
tp_One_ldKL = np.zeros((dim_list_len, repeat))
for j in range(repeat):
    for i in range(dim_list_len):
        dim = dim_list[i]
        dim_p = fKL.pValue_onemeanshift_lineardecreaseKL(samplesize, dim, bootstrapsize= BootstrapSize, iter = iteration)
        # p_One_ldKL[i, :] = dim_p
        tp_One_ldKL[i, j] = fKL.test_power(dim_p, alpha)
        
# np.savetxt("P_ONE_ldKL.csv", p_One_ldKL, delimiter=",")
np.savetxt("BTP_ONE_ldKL.csv", tp_One_ldKL, delimiter=",")



# all mean of Xi shift, meanshift = 1 / n
# p_All_ldKL = np.zeros((dim_list_len, iteration))
tp_All_ldKL = np.zeros((dim_list_len, repeat))
for j in range(repeat):
    for i in range(dim_list_len):
        dim = dim_list[i]
        dim_p = fKL.pValue_allmeanshift_lineardecreaseKL(samplesize, dim, bootstrapsize= BootstrapSize, iter = iteration)
        # p_All_ldKL[i, :] = dim_p
        tp_All_ldKL[i, j] = fKL.test_power(dim_p, alpha)

# np.savetxt("P_ALL_ldKL.csv", p_All_ldKL, delimiter=",")
np.savetxt("BTP_ALL_ldKL.csv", tp_All_ldKL, delimiter=",")



# only mean of X1 shift, meanshift = 1 / n
# p_One_qdKL = np.zeros((dim_list_len, iteration))
tp_One_qdKL = np.zeros((dim_list_len, repeat))
for j in range(repeat):
    for i in range(dim_list_len):
        dim = dim_list[i]
        dim_p = fKL.pValue_onemeanshift_quaddecreaseKL(samplesize, dim, bootstrapsize= BootstrapSize, iter = iteration)
        # p_One_qdKL[i, :] = dim_p
        tp_One_qdKL[i, j] = fKL.test_power(dim_p, alpha)

# np.savetxt("P_ONE_qdKL.csv", p_One_qdKL, delimiter=",")
np.savetxt("BTP_ONE_qdKL.csv", tp_One_qdKL, delimiter=",")



# all mean of Xi shift, meanshift = 1 / i
# p_All_ncKL = np.zeros((dim_list_len, iteration))
tp_All_ncKL = np.zeros((dim_list_len, repeat))
for j in range(repeat):
    for i in range(dim_list_len):
        dim = dim_list[i]
        dim_p = fKL.pValue_allmeanshift_notconstantKL(samplesize, dim, bootstrapsize= BootstrapSize, iter = iteration)
        # p_All_ncKL[i, :] = dim_p
        tp_All_ncKL[i, j] = fKL.test_power(dim_p, alpha)

# np.savetxt("P_ALL_ncKL.csv", p_All_ncKL, delimiter=",")
np.savetxt("BTP_ALL_ncKL.csv", tp_All_ncKL, delimiter=",")



# all mean of Xi shift, meanshift = 0.01
# p_All_liKL = np.zeros((dim_list_len, iteration))
tp_All_liKL = np.zeros((dim_list_len, repeat))
meanvalue = 0.01
for j in range(repeat):
    for i in range(dim_list_len):
        dim = dim_list[i]
        dim_p = fKL.pValue_allmeanshift_linearincreaseKL(samplesize, dim, meanvalue, bootstrapsize= BootstrapSize, iter = iteration)
        # p_All_liKL[i, :] = dim_p
        tp_All_liKL[i, j] = fKL.test_power(dim_p, alpha)

# np.savetxt("P_ALL_liKL.csv", p_All_liKL, delimiter=",")
np.savetxt("BTP_ALL_liKL.csv", tp_All_liKL, delimiter=",")



# only mean of X1 shift, meanshift = 0.1
# p_One_cKL = np.zeros((dim_list_len, iteration))
tp_One_cKL = np.zeros((dim_list_len, repeat))
meanvalue = 0.1
for j in range(repeat):
    for i in range(dim_list_len):
        dim = dim_list[i]
        dim_p = fKL.pValue_onemeanshift_constantKL(samplesize, dim, meanvalue, bootstrapsize= BootstrapSize, iter = iteration)
        # p_One_cKL[i, :] = dim_p
        tp_One_cKL[i] = fKL.test_power(dim_p, alpha)
# np.savetxt("P_ONE_cKL.csv", p_One_cKL, delimiter=",")
np.savetxt("BTP_ONE_cKL.csv", tp_One_cKL, delimiter=",")


