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

p_values = np.zeros((11, iteration))

# under null
tp_KL = np.zeros(dim_list_len)
for i in tqdm(range(dim_list_len)):
    dim = dim_list[i]
    dim_p = h1g.pValue_KLchange(samplesize, dim, "null", meanvalue, bootstrapsize = BootstrapSize, iter = iteration)
    p_values[i] = dim_p
    tp_KL[i] = ksdF.test_power(dim_p, alpha)

np.savetxt("null_p_dim20_iter300.csv", p_values)
np.savetxt("null_TP_dim20_iter300.csv", tp_KL, delimiter=",")



