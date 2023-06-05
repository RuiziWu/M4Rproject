import numpy as np
import A8_functions as A8

samplesize = 200
dim_list = np.array([5, 10, 50, 100, 150, 200, 300])
meanshift = 0.4
b_size = 1000
iteration = 200
set_bandwidth = 'Med'

p_Matrix = A8.pDist_H1_dimchange(samplesize, dim_list, meanshift, set_bandwidth, bootstrapsize = b_size, iter = iteration)

np.savetxt("pDist_H1_05_Med500.csv", p_Matrix, delimiter=",")