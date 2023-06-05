import numpy as np
import A8_functions as A8

samplesize = 100
dim_list = np.array([1, 10, 20, 50, 70, 100])
meanshift = 0.6
b_size = 1000
iteration = 200
set_bandwidth = 'Med'

p_Matrix = A8.pDist_H1_dimchange(samplesize, dim_list, meanshift, set_bandwidth, bootstrapsize = b_size, iter = iteration)

np.savetxt("pDist_H1_5_Med.csv", p_Matrix, delimiter=",")