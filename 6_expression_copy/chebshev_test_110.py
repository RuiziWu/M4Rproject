import numpy as np
import math
from tqdm import tqdm
import A6_functions as A6

samplesize = 100
iteration = 300
meanvalue = 1
alpha = 0.05
count = 6
dim_list = np.array([85, 90, 95, 100, 105, 110])

KSD_value = np.zeros((count, iteration))
KSD_variance = np.zeros((count, iteration))
KSD_prob = np.zeros(count)
KSD_cheb = np.zeros(count)
for i in tqdm(range(count)):
    dim = dim_list[i]
    mean = np.zeros(dim)
    mean[0] = meanvalue
    True_ksd, True_v, Est_ksd, Est_v = A6.comparison_KSD(samplesize, dim, mean, iter = iteration, set_bandwidth = True)
    KSD_value[i, :] = Est_ksd
    KSD_variance[i, :] = Est_v
    KSD_prob[i] = A6.quantile_KSD(Est_ksd, alpha)
    KSD_cheb[i] = True_v / (alpha - True_ksd)**2

np.savetxt("KSD_value_110.csv", KSD_value, delimiter=",")
np.savetxt("KSD_variance_110.csv", KSD_variance, delimiter=",")
np.savetxt("KSD_prob_110.csv", KSD_prob, delimiter=",")
np.savetxt("KSD_cheb_110.csv", KSD_cheb, delimiter=",")

