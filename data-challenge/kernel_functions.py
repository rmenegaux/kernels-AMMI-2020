import numpy as np
from scipy.sparse import issparse

def sparse_to_dense(array):
    return np.array(array.todense()) if issparse(array) else array

def squared_norm(X):
    if not issparse(X):
        return np.sum(X**2, axis=-1)
    else:
        squared_sum = X.multiply(X).sum(-1)
        return np.array(squared_sum).flatten()

def rbf_kernel(X1, X2, sigma=10):
    '''
    Returns the kernel matrix K(X1_i, X2_j): size (n1, n2)
    where K is the RBF kernel with parameter sigma
    
    Input:
    ------
    X1: an (n1, p) matrix
    X2: an (n2, p) matrix
    sigma: float
    '''
    X1_norm = squared_norm(X1)
    X2_norm = squared_norm(X2)
    X1_dot_X2 = sparse_to_dense(np.dot(X1, X2.T))
    gamma = 1 / (2 * sigma ** 2)
    K = np.exp(- gamma * (X1_norm[:, None] + X2_norm[None, :] - 2 * X1_dot_X2))

    return K

def sigma_from_median(X):
    '''
    Returns the median of ||Xi-Xj||
    
    Input
    -----
    X: (n, p) matrix
    '''
    pairwise_diff = X[:, :, None] - X[:, :, None].T
    pairwise_diff *= pairwise_diff
    euclidean_dist = np.sqrt(pairwise_diff.sum(axis=1))
    return np.median(euclidean_dist)
    
def linear_kernel(X1, X2):
    '''
    Returns the kernel matrix K(X1_i, X2_j): size (n1, n2)
    where K is the linear kernel
    
    Input:
    ------
    X1: an (n1, p) matrix
    X2: an (n2, p) matrix
    '''
    K = X1.dot(X2.T)
    return sparse_to_dense(K)

def polynomial_kernel(X1, X2, degree=2):
    '''
    Returns the kernel matrix K(X1_i, X2_j): size (n1, n2)
    where K is the quadratic kernel
    
    Input:
    ------
    X1: an (n1, p) matrix
    X2: an (n2, p) matrix
    '''
    return (1 + linear_kernel(X1, X2))**degree

def mismatch_kernel(X1, X2):
    '''
    Returns the kernel matrix K(X1_i, X2_j): size (n1, n2)
    where K is the (k,1)-mismatch kernel
    
    Input:
    ------
    X1: tuple
       - X1[0] is an (n1, k * 4**(k-1)) matrix containing the
         (k, 1)-mismatch feature representation
       - X1[1] is an (n1, 4**k) matrix containing the
         exact match k-mer representation
    X2: same as X1
    '''
    # Deduce k-mer size from number of features
    kmer_size = (4 * X1[0].shape[1]) / (X1[1].shape[1])
    mismatches = linear_kernel(X1[0], X2[0])
    exact_matches = linear_kernel(X1[1], X2[1])
    
    return mismatches - (kmer_size - 1) * exact_matches