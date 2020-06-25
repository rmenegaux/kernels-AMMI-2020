import numpy as np
from kernel_base import KernelMethodBase


class KernelRidgeRegression(KernelMethodBase):
    '''
    Kernel Ridge Regression
    '''
    def __init__(self, lambd=0.1, **kwargs):
        self.lambd = lambd
        super().__init__(**kwargs)

    def fit_K(self, K, y):
        '''
        K: Option to pass in kernel matrix directly
        '''
        self.y_train = y
        n = len(self.y_train)
        
        A = K + self.lambd * n * np.eye(n)
        # self.alpha = (K + n lambda I)^-1 y
        self.alpha = np.linalg.solve(A , self.y_train)

        return self
    
    def decision_function_K(self, K):
        return K.dot(self.alpha)
    
    def predict(self, X):
        return self.decision_function(X)
    
    def predict_K(self, K):
        return self.decision_function_K(K)
    
    
class WeightedKernelRidgeRegression(KernelRidgeRegression):
    '''
    Weighted Kernel Ridge Regression
    
    This is just used for the KernelLogistic following up
    '''
    def fit_K(self, K, y, sample_weights=None):

        self.y_train = y
        n = len(self.y_train)
        
        w = np.ones_like(self.y_train) if sample_weights is None else sample_weights
        W = np.diag(np.sqrt(w))
        
        A = W.dot(K).dot(W)
        A[np.diag_indices_from(A)] += self.lambd * n
        # self.alpha = W (K + n lambda I)^-1 W y
        self.alpha = W.dot(np.linalg.solve(A , W.dot(self.y_train)))

        return self


class KernelRidgeClassifier(KernelRidgeRegression):
    '''
    Kernel Ridge Classification
    '''
    def predict(self, X):
        return np.where(self.decision_function(X) < 0, -1, 1)

    def predict_K(self, K):
        return np.where(self.decision_function_K(K) < 0, -1, 1)


def sigmoid(x):
    # tanh version helps avoid overflow problems
    return .5 * (1 + np.tanh(.5 * x))
    # return 1 / (1 + np.exp(-x))


class KernelLogisticRegression(KernelRidgeClassifier):
    '''
    Kernel Logistic Regression
    '''
    def fit_K(self, K, y, max_iter=100, tol=1e-5):

        self.y_train = y
                
        # IRLS
        WKRR = WeightedKernelRidgeRegression(
            lambd=2*self.lambd,
            kernel=self.kernel_name,
            **self.kernel_parameters
        )
        # Initialize
        alpha = np.zeros_like(y)
        # Iterate until convergence or max iterations
        for n_iter in range(max_iter):
            alpha_old = alpha
            f = K.dot(alpha_old)
            w = sigmoid(f) * sigmoid(-f)
            z = f + y / sigmoid(y*f)
            alpha = WKRR.fit_K(K, z, sample_weights=w).alpha
            # Break condition (achieved convergence)
            if np.sum((alpha-alpha_old)**2) < tol:
                break
        self.n_iter = n_iter
        self.alpha = alpha

        return self
            
    def decision_function_K(self, K):
        return 2*sigmoid(K.dot(self.alpha)) - 1