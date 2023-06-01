import numpy as np
import A8_functions as A8

samplesize = 100
dim_list = np.array([30])
b_size = 1000
iteration = 200

p_Matrix, S_Matrix = A8.pDist_H0_dimchange(samplesize, dim_list, set_bandwidth = 1., bootstrapsize = b_size, iter = iteration)

np.savetxt("test.csv", p_Matrix, delimiter=",")
np.savetxt("testS.csv", S_Matrix, delimiter=",")
