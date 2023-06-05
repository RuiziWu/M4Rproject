import numpy as np
import A8_theo_func as ATheo

samplesize = 100
dim_list = np.array([5, 10, 50, 100, 150])
meanshift = 5
b_size = 1000
iteration = 400
alpha = 0.05
beta = 0.05

p_Matrix = ATheo.pDist_nvsdim(dim_list, meanshift, alpha, beta, bootstrapsize = b_size, iter = iteration)

np.savetxt("pDist_nvsdim_m5.csv", p_Matrix, delimiter=",")