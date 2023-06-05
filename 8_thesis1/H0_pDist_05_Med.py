import numpy as np
import A8_functions as A8


dim_list = np.array([1, 10, 50, 100, 500, 1000, 1500])
meanshift = 0
b_size = 1000
iteration = 400
set_bandwidth = 'Med'

samplesize = 150
p_Matrix = A8.pDist_H1_dimchange(samplesize, dim_list, meanshift, set_bandwidth, bootstrapsize = b_size, iter = iteration)
np.savetxt("pDist_H0_05_Med150.csv", p_Matrix, delimiter=",")

samplesize = 100
p_Matrix = A8.pDist_H1_dimchange(samplesize, dim_list, meanshift, set_bandwidth, bootstrapsize = b_size, iter = iteration)
np.savetxt("pDist_H0_05_Med100.csv", p_Matrix, delimiter=",")

samplesize = 50
p_Matrix = A8.pDist_H1_dimchange(samplesize, dim_list, meanshift, set_bandwidth, bootstrapsize = b_size, iter = iteration)
np.savetxt("pDist_H0_05_Med50.csv", p_Matrix, delimiter=",")