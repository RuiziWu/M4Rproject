import numpy as np
import A8_functions as A8

samplesize = 1000
dim_list = np.array([1, 10, 20, 30, 40, 50, 60])
b_size = 1500
iteration = 400

p_Matrix = A8.pDist_H0_dimchange(samplesize, dim_list, bootstrapsize = b_size, iter = iteration, set_bandwidth = True)

np.savetxt("pDist_H0_smalldim.csv", p_Matrix, delimiter=",")