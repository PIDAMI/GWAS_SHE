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

# with_bias_column checks if covariates matrix should include column of ones corresponding to bias term
def gen_random_data(seed: int, n: int, m: int, k: int, with_bias_column = True) -> tuple():
    np.random.seed(seed)
    S = np.random.normal(0, 1, size=(n, m)).astype(np.float64) # SNP matrix
    X = np.random.uniform(0, 1, size=(n, k)) # covariates matrix 
    if with_bias_column:
        ones = np.ones(n)
        X = np.append(ones.reshape(n, 1), X, axis=1).astype(np.float64) # covariates matrix concatenated with const term in regression

    y = np.random.binomial(1, 0.5, n).reshape(n,1).astype(np.float64)
    return (y,S,X)
