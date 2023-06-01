import numpy as np
from tqdm import tqdm
import A7_functions as A7

samplesize = 3000
iteration = 100
meanvalue = 1 # first component with mean_shift = 1
dim_list = np.array([1, 10, 50, 100, 500, 1000])
count = len(dim_list)


True_variance = np.zeros(count)
Exp_variance = np.zeros(count)
for i in tqdm(range(count)):
    dim = dim_list[i]
    mean = np.zeros(dim)
    mean[0] = meanvalue
    True_variance[i] = A7.True_Variance(samplesize, dim, mean)
    Exp_variance[i] = A7.Emp_Variance(samplesize, dim, mean, iteration=iteration)

np.savetxt("True_variance_iter300_sample3000.csv", True_variance, delimiter=",")
np.savetxt("Emp_variance_iter300_sample3000.csv", Exp_variance, delimiter=",")