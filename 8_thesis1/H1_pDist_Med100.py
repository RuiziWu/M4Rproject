import numpy as np
import A8_functions as A8

samplesize = 100
dim_list = np.array([5, 10, 50, 100, 150, 200, 300])
meanshift1 = 0.42
meanshift2 = 0.45
meanshift3 = 0.47

b_size = 1000
iteration = 400
set_bandwidth = 'Med'

p_Matrix1 = A8.pDist_H1_dimchange(samplesize, dim_list, meanshift1, set_bandwidth, bootstrapsize = b_size, iter = iteration)
p_Matrix2 = A8.pDist_H1_dimchange(samplesize, dim_list, meanshift2, set_bandwidth, bootstrapsize = b_size, iter = iteration)
p_Matrix3 = A8.pDist_H1_dimchange(samplesize, dim_list, meanshift3, set_bandwidth, bootstrapsize = b_size, iter = iteration)

np.savetxt("pDist_H1_Med42R.csv", p_Matrix1, delimiter=",")
np.savetxt("pDist_H1_Med45R.csv", p_Matrix2, delimiter=",")
np.savetxt("pDist_H1_Med47R.csv", p_Matrix3, delimiter=",")
