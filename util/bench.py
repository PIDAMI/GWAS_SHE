import numpy as np
import time

"""
model - instance of model to benchmark
num_runs - number of benchmarks
n,m,k - problem size(same as in SNPLogRegression)
"""
def bench_model(model, num_runs:int, n:int, m:int, k:int) -> list[dict]:
    results = {"elapsed_time": list(), "msip": list(), }
    
    for i in range(num_runs):
        np.random.seed(i)
        S = np.random.normal(0, 1, size=(n, m)) # SNP matrix
        X0 = np.random.uniform(0, 1, size=(n, k)) # covariates matrix 
        ones = np.zeros(n) + 1
        X = np.append(ones.reshape(n, 1), X0, axis=1) # covariates matrix concatenated with const term in regression
        y = np.random.binomial(1, 0.5, n).reshape(n,1)

        before = time.time()
        model.fit(y,S,X)
        elapsed_time = time.time() - before
        msip = 1e-06 * n * m / elapsed_time
        results["elapsed_time"].append(elapsed_time)
        results["msip"].append(msip)
    
    return results



