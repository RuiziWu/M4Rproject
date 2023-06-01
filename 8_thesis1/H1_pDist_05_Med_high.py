import numpy as np
import A8_functions as A8

samplesize = 1000
dim_list = np.array([1, 10, 50, 100, 500, 1000, 1500])
meanshift = 0.5
b_size = 1000
iteration = 200
set_bandwidth = 'Med'

p_Matrix = A8.pDist_H1_dimchange(samplesize, dim_list, meanshift, set_bandwidth, bootstrapsize = b_size, iter = iteration)

np.savetxt("pDist_H1_05_Med_high.csv", p_Matrix, delimiter=",")