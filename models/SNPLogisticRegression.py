import numpy as np
import statsmodels.api as sm
import scipy.stats as stat

class SNPLogRegression:
    
    """
    y - predicting value, array of bools, dim(y) = 
    S - matrix of SNPs, m for each individual, dim(S) = (n,m)
    X - matrix of covariates, k for each individual, dim(X) = (n,k)
    """
    def __init__(self) -> None:
        pass
         
    def fit(self, y : np.ndarray, S : np.ndarray, X : np.ndarray) -> None:
        self.y = y
        self.X = X
        self.S = S

        #TODO: check if all dimensions of y,X,S are coherent
        self.n = np.shape(y)[0]
        self.m = np.shape(S)[1]
        self.k = np.shape(X)[1]
        
        self.pvalues = np.zeros(self.m)
        self.std_err = np.zeros(self.m)
        self.coef = np.zeros(self.m)

        self._fit()
        

class SemiParallelSNPLogRegression(SNPLogRegression):
    
    def _fit(self) -> None:
        model = sm.GLM(self.y, self.X, family=sm.families.Binomial()).fit()
        p = model.fittedvalues.reshape(self.n,1)

        w = p * (1 - p) # diag matrix with const elem, represented as an array
        working_var = np.log(p / (1 - p)) + (self.y - p) / (p * (1 - p)) # denoted as z 
        X_transposed_w = (self.X * w).T # X^T * W

        # find betta_t, needed to find z*(working variable for transformed problem w/t covariate)
        betta_t = np.linalg.solve(X_transposed_w @ self.X, X_transposed_w @ working_var) 
        working_var_trans = working_var - self.X @ betta_t # denoted as z*
        U = np.linalg.solve(X_transposed_w @ self.X, X_transposed_w @ self.S) 
        S_star = self.S - self.X @ U # transformed S
        
        # now we transformed problem to one w/t covariates and can solve as usual
        Str2 = np.sum(w * S_star**2, axis = 0) # m size vector of sums(w_i * s_i^2)
        self.coef = np.sum(working_var_trans * w * S_star, axis=0) / Str2 # m size vect of sums(wi zi si) / str2, denoted as betta
        self.std_err = Str2**-0.5 
        self.pval = 2 * stat.norm.cdf(-np.abs(self.coef / self.std_err))


class BruteForceSNPLogRegression(SNPLogRegression):

    def _fit(self) -> None:
        for i in range(self.m):
            X_plus_si = np.append(self.S[:,i].reshape(self.n, 1), self.X, axis=1)
            model = sm.GLM(self.y, X_plus_si, family=sm.families.Binomial()).fit()
            self.coef[i] = model.params[0]
            # self.std_err[i] = model.err[0]
            self.pvalues[i] = model.pvalues[0]
