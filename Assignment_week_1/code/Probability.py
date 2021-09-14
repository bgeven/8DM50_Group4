#Code for the class-conditional probability
#Group: 4, Course: 8DM50

import matplotlib.pyplot as plt
import numpy as np

def Euclidean_distance(x,y):
    """function that calculates the squared euclidean distance between two points
    input: x (vector, containing multiple dimensions)
    , y: (vector, containg number dimensions equal to x)
    output: d(float, distance between point x and y)
    """
    d = 0 #starting value

    for j in range(len(y)): #looping through the dimension coordinates of y
        d += (y[j] - x[j])**2 #calculating the distance according to the euclidean distance
        
    return(d)

#Split dataset in Malignant and Benign part
def splitting_data(X,Y):
    """Function splits the dataset based on given markers for Malignant or Benign
    X: has to be a numpy array with the same size as Y, but 2 dimensional from itself
    Y: has to be a numpy array with the same size as X for its first dimension, but made as a 2 dimensional array
    containing only zeros and ones
    returns:
    X_yes: list of values from X that correspond to a 0 (Malignant) in Y
    Y_no: list of values from Y that correspond to a 1 (Benign) in Y
    """
    mal = np.where(Y==0)
    ben = np.where(Y==1)
    X_mal = X[mal,:][0]
    X_ben = X[ben,:][0]
    
    return X_mal, X_ben

#function for probability
def normpdf(x, mean, sd):
    """This function calculates the probability of a 1D array.
    x: a 1D array
    mean: the mean of the array
    sd: the standard deviation of the array
    returns: the probability density of the input array
    """
    prob_density = (np.pi*sd) * np.exp(-0.5* ((x-mean)/sd)**2)
    return prob_density

#calculate probability of X
def prob_per_marker(X):
    """In this function the normal distributed probability is calculated for an array with 2 dimensions.
    X: array with 2 dimensions containing the data points for markers
    returns: a 2D array with the probability for each marker
    """
    pdf_X = []
    for column_number in range(len(X[1])):
        mean_x = np.mean(X[:,column_number])
        std_x = np.std(X[:,column_number])
        pdf_x = normpdf(X[:,column_number],mean_x,std_x)
        pdf_X.append(pdf_x)
    return np.array(pdf_X)

def plotting_adj(X_mal, X_ben, P_mal, P_ben):
    """In this function the class-conditional probability is plotted.
    X: The data for the conditions
    Y: The condition, either 1 or 0
    P: the class-conditional probability
    """
    fig, axs = plt.subplots(5,6,figsize=(15, 12))
    axs = axs.flatten()
    
    for i in range(0,len(P_mal)):
        axs[i].scatter(X_mal[:,i], np.transpose(P_mal[i]), label='Malignant',color='blue')
        axs[i].scatter(X_ben[:,i], np.transpose(P_ben[i]), label='Benign',color='red')
    plt.show()