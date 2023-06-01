import numpy as np
from tqdm import tqdm
import A7_functions as A7

samplesize = 5000
meanvalue = 1 # first component with mean_shift = 1
dim_list = np.array([1, 10, 50, 100, 500, 1000])
count = len(dim_list)


True_ksd = np.zeros(count)
Emp_ksd = np.zeros(count)
for i in tqdm(range(count)):
    dim = dim_list[i]
    mean = np.zeros(dim)
    mean[0] = meanvalue
    True_ksd[i] = A7.True_KSD(mean, dim)
    Emp_ksd[i] = A7.Emp_Expectation(samplesize, dim, mean)

np.savetxt("True_ksd_dim1000_sample5000.csv", True_ksd, delimiter=",")
np.savetxt("Emp_ksd_dim1000_sample5000.csv", Emp_ksd, delimiter=",")
