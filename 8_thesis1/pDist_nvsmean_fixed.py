import numpy as np
import A8_functions as A8
from tqdm import tqdm

samplesize = 100
dim_list = np.array([100])
mean_list = np.array([0.42, 0.45, 0.5])
b_size = 1000
iteration = 100
set_bandwidth = 'Med'

p_Matrix = np.zeros((3, iteration))

for i in tqdm(range(3)):
    meanshift = mean_list[i]
    p_Matrix[i, :] = A8.pDist_H1_dimchange(samplesize, dim_list, meanshift, set_bandwidth, bootstrapsize = b_size, iter = iteration)

np.savetxt("pDist_nvsmean_fixed.csv", p_Matrix, delimiter=",")