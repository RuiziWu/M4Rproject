import numpy as np
from tqdm import tqdm
import A5_functions as A5

samplesize = 1000
iteration = 300
meanvalue = 1
alpha = 0.05
count = 2
dim_list = np.array([i*5 for i in range(count)])
dim_list[0] = 1
bandwidth = 0.1

# KSD_dim_change = np.zeros((count, iteration))
# KSD_prob = np.zeros(count)
# KSD_cheb = np.zeros(count)
# for i in tqdm(range(count)):
#     dim = dim_list[i]
#     mean = np.zeros(dim)
#     mean[0] = meanvalue
#     KSD_i = A5.KSD_values(samplesize, dim, mean, iter = iteration, MH_method = False, set_bandwidth = bandwidth)
#     KSD_dim_change[i, :] = KSD_i

# np.savetxt("KSD_dim5_bandwidth.csv", KSD_dim_change, delimiter=",")

KSD_dim_change = np.zeros((count, iteration))
KSD_prob = np.zeros(count)
KSD_cheb = np.zeros(count)
for i in tqdm(range(count)):
    dim = dim_list[i]
    mean = np.zeros(dim)
    mean[0] = meanvalue
    KSD_i = A5.KSD_values(samplesize, dim, mean, iter = iteration, MH_method = False, set_bandwidth = bandwidth)
    KSD_dim_change[i, :] = KSD_i
    prob_alpha = A5.quantile_KSD(KSD_i, alpha)
    KSD_prob[i] = prob_alpha
    expectation = A5.True_KSD(mean, dim, bandwidth)
    variance = A5.Var_KSD(samplesize, mean, dim, bandwidth)
    cheb_bound = variance / (alpha - expectation)**2
    KSD_cheb[i] = cheb_bound

np.savetxt("KSD_dim.csv", KSD_dim_change, delimiter=",")
np.savetxt("KSD_prob.csv", KSD_prob, delimiter=",")
np.savetxt("KSD_cheb.csv", KSD_cheb, delimiter=",")
