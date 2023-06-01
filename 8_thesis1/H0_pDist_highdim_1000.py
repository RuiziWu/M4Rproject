import numpy as np
import A8_functions as A8

samplesize = 1000
dim_list = np.array([1, 10, 50, 100, 500, 1000, 1500])
b_size = 1000
iteration = 100

p_Matrix = A8.pDist_H0_dimchange(samplesize, dim_list, bootstrapsize = b_size, iter = iteration, set_bandwidth = 'Med')

np.savetxt("pDist_H0_dim_1000.csv", p_Matrix, delimiter=",")