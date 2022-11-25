import numpy as np
import copy
from Pyfhel import Pyfhel

class CKKS():
    def __init__(self, params, n_samples, num_snp, num_covariates) -> None:
        self.he = Pyfhel()  
        self.he.contextGen(**params)
        self.he.keyGen()
        self.he.rotateKeyGen()
        self.he.relinKeyGen()

        self.mod = params["n"]
        self.n = n_samples
        self.m = num_snp
        self.k = num_covariates
    
    # size is number of non-zero elements of ciphertext(as its size is mod/2 
    # and we typically use vectors of less size, non used components are fileld with 0)
    def replicate(self, ctx, size):
        one = np.zeros(self.mod,dtype=np.float64)
        res = list()
        for i in range(size):
            one[i] = 1
            tmp = ctx * one
            self.he.relinearize(tmp)
            one[i] = 0
            for j in range(np.floor(np.log2(self.mod)) - 1):
                tmp += self.he.rotate(tmp, k=2**j, in_new_ctxt=True)
            self.he.relinearize(tmp)
            res.append(tmp)
        return res

    # matrices are represented as lists of vectors as CKKS cant handle matrices as they are
    def mat_vec_mult(self,matr,vec,size):
        column_repr = self.replicate(vec,size)
        res = matr[0] * column_repr[0]
        self.he.relinearize(res)
        for i in range(1,size):
            res += matr[i] * column_repr[i]
            self.he.relinearize(res)
        return res

    # size for m2; [m2] = (i,j) => m2 is list of j vectors, representing its columns, size == i 
    def mat_mat_mult(self,m1,m2,size):
        res = self.mat_vec_mult(m1,m2[0],size)
        for i in range(1, len(m2)):
            res += self.mat_vec_mult(m1,m2[i],size)
        return res


    # vector of sums of components of each column, e.g. col_sum([[1,2,3],[4,5,6]])=[6,15] as matrices represented as lists of columns in our framework
    # returns list of vectors, each vector has all components filled with needed sum of corresponding column
    # e.g. col_sum([[1,2,3],[4,5,6]])=[[6,6],[15,15]]
    # size is length of columns
    def column_sums(self, matr):
        res = list()
        for i in range(len(matr)):
            sum_i = copy.deepcopy(matr[i])
            for j in range(np.floor(np.log2(self.mod)) - 1):
                sum_i += self.he.rotate(sum_i, k=2**j, in_new_ctxt=True)
            res.append(sum_i)
        return res

    # compute sigmoid function of ciphertext(which is vector) component-wise
    # from Kim et al.Logistic Regression Model Training based on the Approximate Homomorphic Encryption
    def sigmoid(self, vec, size):
        const_term = self.he.encrypt(np.array([0.5]*size, dtype=np.float64))
        pow_1 = vec
        pow_2 = pow_1 * pow_1
        self.he.relinearize(pow_2)
        self.he.rescale_to_next(pow_2)
        pow_3 = pow_1 * pow_2
        self.he.relinearize(pow_3)
        self.he.rescale_to_next(pow_3)
        # pow_4 = pow_2 * pow_2
        # self.he.relinearize(pow_4)
        # self.he.rescale_to_next(pow_4)
        # pow_5 = pow_4 * pow_1
        # self.he.relinearize(pow_5)
        # self.he.rescale_to_next(pow_5)
        # pow_6 = pow_4 * pow_2
        # self.he.relinearize(pow_6)
        # self.he.rescale_to_next(pow_6)
        # pow_7 = pow_6 * pow_1
        # self.he.relinearize(pow_7)
        # self.he.rescale_to_next(pow_7)
        res = const_term 
        + self.he.encrypt(np.array([1.73496 / 8]*size, dtype=np.float64)) * pow_1  
        + self.he.encrypt(np.array([4.19407 / 8**3]*size, dtype=np.float64)) * pow_3
        # - self.he.encrypt(np.array([5.43402 / 8**5]*size, dtype=np.float64)) * pow_5
        # + self.he.encrypt(np.array([2.50739 / 8**7]*size, dtype=np.float64)) * pow_7
        self.he.relinearize(res)
        self.he.rescale_to_next(res)
        return res

