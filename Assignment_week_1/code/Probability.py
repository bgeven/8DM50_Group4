#Code for the class-conditional probability
#Group: 4, Course: 8DM50

import matplotlib.pyplot as plt
import numpy as np

def splitting_data(X,Y):
    """
    Function that splits the dataset based on given markers for targets 0 and 1
    param X: numpy array with the same size as Y, containing features
    param Y: numpy array with the same size as X, containing only zeros and ones
    return X_zero: matrix, with values from X that correspond to 0 in Y
    return X_one: matrix, with values from Y that correspond to 1 in Y
    """
    
    zero = np.where(Y==0)       
    one = np.where(Y==1)
    X_zero = X[zero,:][0]
    X_one = X[one,:][0]
    
    return X_zero, X_one

def normpdf(x, mean, sd):
    """
    Function that calculates the probability with Gaussian distribution of an 1D array
    param x: 1D array
    param mean: average of the 1D array
    param sd: standard deviation of the 1D array
    return prob_density: the probability density of the input array
    """
   
    prob_density = (np.pi*sd) * np.exp(-0.5* ((x-mean)/sd)**2)
    
    return prob_density

def prob_per_marker(X):
    """
    Function that calculates the normal distributed probability for an array with 2 dimensions
    param X: array with 2 dimensions containing the data points for markers
    returns: a 2D array with the probability for each marker
    """
    
    pdf_X = []  # create empty list to store the probability values
    
    for column_number in range(len(X[1])):
        mean_x = np.mean(X[:,column_number])                # calculate mean values of specific feature
        std_x = np.std(X[:,column_number])                  # calculate standard deviation of specific feature
        pdf_x = normpdf(X[:,column_number],mean_x,std_x)    # calculate the probability with Gaussian distribution
        pdf_X.append(pdf_x)
        
    return np.array(pdf_X)

def plotting(X_0, X_1, P_0, P_1):
    """
    Function that plots the class-conditional probability for the features, split by marker
    param X_0: matrix, with values from X that correspond to 0 in Y
    param X_1: matrix, with values from X that correspond to 1 in Y
    param P_0: probability for marker '0'
    param P_1: probability for marker '1'
    """
    
    fig, axs = plt.subplots(5,6,figsize=(15, 12))
    axs = axs.flatten()
    
    for i in range(0,len(P_0)):
        axs[i].scatter(X_0[:,i], np.transpose(P_0[i]), label='Malignant',color='blue')
        axs[i].scatter(X_1[:,i], np.transpose(P_1[i]), label='Benign',color='red')
        
    plt.show()