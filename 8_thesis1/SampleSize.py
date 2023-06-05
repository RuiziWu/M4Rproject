import numpy as np
from tqdm import tqdm
import A8_theo_func as A8_func

dim_list = np.array([5, 10, 50, 100, 150, 200, 300])
mean_list = np.array([0.4, 0.42, 0.45, 0.47, 0.5, 0.52, 0.55, 0.57])
alpha = 0.1
beta = 0.1

dimcount = len(dim_list)
meancount = len(mean_list)

ndMatrix = np.zeros((meancount, dimcount))
nmMatrix = np.zeros((meancount, dimcount))

for i in tqdm(range(meancount)):
    mean = mean_list[i]
    for j in range(dimcount):
        dim = dim_list[j]
        ndMatrix[i, j] = A8_func.LowerBound_nvsdim(dim, mean, alpha, beta)
        nmMatrix[i, j] = A8_func.LowerBound_nvsmean(dim, mean, alpha, beta)


np.savetxt("ndMatrix.csv", ndMatrix, delimiter=",")
np.savetxt("nmMatrix.csv", nmMatrix, delimiter=",")
