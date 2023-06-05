import numpy as np
import A8_theo_func as ATheo

dim = 100
b_size = 1000
iteration = 100
alpha = 0.1
beta = 0.1

mean_list = np.array([0.42])
p_Matrix = ATheo.pDist_nvsmean(dim, mean_list, alpha, beta, bootstrapsize = b_size, iter = iteration)

np.savetxt("pDist_nvsmean_042.csv", p_Matrix, delimiter=",")


mean_list = np.array([0.45])
p_Matrix = ATheo.pDist_nvsmean(dim, mean_list, alpha, beta, bootstrapsize = b_size, iter = iteration)

np.savetxt("pDist_nvsmean_045.csv", p_Matrix, delimiter=",")


mean_list = np.array([0.5])
p_Matrix = ATheo.pDist_nvsmean(dim, mean_list, alpha, beta, bootstrapsize = b_size, iter = iteration)

np.savetxt("pDist_nvsmean_050.csv", p_Matrix, delimiter=",")