import numpy as np
import A8_functions2 as A8

samplesize = 1000
dim_list = np.array([1, 10, 50, 100, 500, 1000, 1500])
meanshift = 0.5
b_size = 1000
iteration = 200
set_bandwidth = 1

p_Matrix = A8.pValue_allmeanshift(samplesize, dim_list, meanshift, set_bandwidth, bootstrapsize = 1000, iter = iteration)

np.savetxt("pDist_H1_05_bd1_all_high.csv", p_Matrix, delimiter=",")