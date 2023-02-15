import numpy as np
from tqdm import tqdm
import A5_functions as A5

samplesize = 1000
iteration = 500
meanvalue = 1
bandwidth = 0.1
count = 11
dim_list = np.array([i*2 for i in range(count)])
dim_list[0] = 1

KSD_dim_change = np.zeros(count)
True_KSD = np.zeros(count)
for i in tqdm(range(count)):
    dim = dim_list[i]
    mean = np.zeros(dim)
    mean[0] = meanvalue * (dim ** (1 / 6))
    mean_KSD = A5.E_KSD(samplesize, dim, mean, iter = iteration, MH_method = False, set_bandwidth = bandwidth)
    KSD_dim_change[i] = mean_KSD
    True_KSD[i] = (bandwidth ** dim) * ((bandwidth ** 2 + 2) ** (- dim / 2)) * np.dot(mean, mean)

np.savetxt("KSD_dim_change_3incre.csv", KSD_dim_change, delimiter=",")
np.savetxt("True_KSD_3incre.csv", True_KSD, delimiter=",") 
