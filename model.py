import numpy
from scipy.optimize import fmin_l_bfgs_b
from sklearn.linear_model import LogisticRegression

def sigmoid(z):
    return 1 / (1 + numpy.exp(-z))

def logreg_obj(wb, D, L, reg_lambda=0.0):
    w = wb[:-1]
    b = wb[-1]
    z = numpy.dot(w.T, D) + b
    s = sigmoid(z)
    
    # Compute log-loss
    loss = -numpy.mean(L * numpy.log(s + 1e-10) + (1 - L) * numpy.log(1 - s + 1e-10))
    # Add L2 regularization
    loss += 0.5 * reg_lambda * numpy.sum(w ** 2)
    return loss

def logreg_grad(wb, D, L, reg_lambda=0.0):
    w = wb[:-1]
    b = wb[-1]
    z = numpy.dot(w.T, D) + b
    s = sigmoid(z)
    
    grad_w = numpy.dot(D, (s - L).T) / D.shape[1] + reg_lambda * w
    grad_b = numpy.mean(s - L)
    return numpy.append(grad_w, grad_b)


def train_logistic_regression(DTR, LTR, reg_lambda=0.0):
    init_wb = numpy.zeros(DTR.shape[0] + 1)
    result = fmin_l_bfgs_b(
        func=logreg_obj,
        x0=init_wb,
        fprime=logreg_grad,
        args=(DTR, LTR, reg_lambda)
    )
    return (result[0][:-1], result[0][-1])  # return w and b


# will NOT be used
def train_using_library(DTR, LTR, reg_lambda=0.001):
    DTR_T = DTR.T
    model = LogisticRegression(penalty='l2', C=1/reg_lambda)
    model.fit(DTR_T, LTR)
    return model.coef_[0], model.intercept_[0]

def train_logistic_regression_sklearn(DTR, LTR, reg_lambda=0.001):
    """
    Train logistic regression using scikit-learn's LogisticRegression.
    DTR: features (features x samples)
    LTR: labels (samples,)
    reg_lambda: regularization strength (higher means more regularization)
    Returns: (w, b)
    """
    DTR_T = DTR.T
    model = LogisticRegression(penalty='l2', C=1/reg_lambda, solver='lbfgs', max_iter=1000)
    model.fit(DTR_T, LTR)
    return model.coef_[0], model.intercept_[0]

from sklearn.svm import SVC

def train_svm(DTR, LTR, C=1.0, kernel='linear', gamma='scale'):
    """
    Train an SVM classifier using scikit-learn's SVC.
    DTR: features (features x samples)
    LTR: labels (samples,)
    C: regularization parameter
    kernel: kernel type ('linear', 'rbf', etc.)
    gamma: kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
    Returns: trained SVC model
    """
    DTR_T = DTR.T
    model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
    model.fit(DTR_T, LTR)
    return model
