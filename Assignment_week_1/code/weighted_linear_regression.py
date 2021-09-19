#Code for the weighted linear regression
#Group: 4, Course: 8DM50

import numpy as np

def wlsq(X, y, d):
    """
    Function that determines the Weighted least squares linear regression
    :param X: matrix, containing input data
    :param y: vector, containing targets
    :param d: vector, containing weights
    :return beta: vector, containing estimated coefficients for the linear regression
    """
    
    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # calculate the coefficients for weighted linear regression
    beta = np.dot(np.linalg.inv(np.dot(np.dot(X.T,d),X)),np.dot(np.dot(X.T,d),y))
    
    return beta