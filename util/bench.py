import numpy as np
import time

"""
model - instance of model to benchmark
num_runs - number of benchmarks
n,m,k - problem size(same as in SNPLogRegression)
"""
def bench_model(model, num_runs:int, n:int, m:int, k:int) -> dict:
    results = {"elapsed_time": np.zeros(num_runs), "msip": np.zeros(num_runs), }
    
    for i in range(num_runs):
        y, S, X = gen_random_data(i, n, m, k)

        before = time.time()
        model.fit(y,S,X)
        elapsed_time = time.time() - before
        msip = 1e-06 * n * m / elapsed_time
        
        results["elapsed_time"][i] = elapsed_time
        results["msip"][i] = msip
    
    return results


def gen_random_data(seed: int, n: int, m: int, k: int) -> tuple():
    np.random.seed(seed)
    S = np.random.normal(0, 1, size=(n, m)).astype(np.float64) # SNP matrix
    X0 = np.random.uniform(0, 1, size=(n, k)) # covariates matrix 
    ones = np.zeros(n) + 1
    X = np.append(ones.reshape(n, 1), X0, axis=1).astype(np.float64) # covariates matrix concatenated with const term in regression
    y = np.random.binomial(1, 0.5, n).reshape(n,1).astype(np.float64)
    return (y,S,X)
