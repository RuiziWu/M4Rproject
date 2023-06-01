import numpy as np
import A8_functions as A8

samplesize = 100
dim_list = np.array([1, 10, 50, 100, 500, 1000, 1500])
b_size = 1000
iteration = 100

p_Matrix = A8.pDist_H0_dimchange(samplesize, dim_list, bootstrapsize = b_size, iter = iteration, set_bandwidth = 1)

np.savetxt("pDist_H0_dim_100_bd1.csv", p_Matrix, delimiter=",")