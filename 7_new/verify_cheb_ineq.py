import numpy as np
from tqdm import tqdm
import A7_functions as A7


samplesize = 1000
iteration = 300
meanvalue = 1 # first component with mean_shift = 1
dim_list = np.array([1, 10, 50, 100, 500, 1000])
count = len(dim_list)
# count = 3
# dim_list = np.array([i*2 for i in range(count)])
# dim_list[0] = 1


True_ksd = np.zeros(count)
True_variance = np.zeros(count)
P_ksd_alpha = np.zeros(count)
KSD_whole = np.zeros((count, iteration))
for i in tqdm(range(count)):
    dim = dim_list[i]
    mean = np.zeros(dim)
    mean[0] = meanvalue
    t_ksd = A7.True_KSD(mean, dim)
    True_ksd[i] = t_ksd
    True_variance[i] = A7.True_Variance(samplesize, dim, mean)
    ksd_i = A7.Unbiased_KSDs(samplesize, dim, mean, iteration=iteration)
    KSD_whole[i, :] = ksd_i
    alpha = t_ksd + 0.05
    P_ksd_alpha[i] = A7.quantile_KSD(ksd_i, alpha)

np.savetxt("True_expectation_test.csv", True_ksd, delimiter=",")
np.savetxt("True_variance_test.csv", True_variance, delimiter=",")
np.savetxt("P_ksd_test.csv", P_ksd_alpha, delimiter=",")
np.savetxt("KSD_whole_test.csv", KSD_whole, delimiter=",")