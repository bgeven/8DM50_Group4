#Code for the k-NN classification and k-NN regression
#Group: 4, Course: 8DM50

import numpy as np

def euclidean_distance(X_test, X_train):
    """
    Function that calculates the Euclidean distance between two points
    param X_test: matrix, containing data from test set
    param X_train: matrix, containing data from training set
    return euclidean_matrix: matrix, with distances between matrices with test en training data
    """
    
    euclidean_matrix = None
    
    for i in range(len(X_test)):        
        x_test = X_test[i]            # one instance
        euclidean_distances = []      # list for distances of instance to training data instances
        
        for j in range(len(X_train)):       # loop over the instances in the training data
            x_train = X_train[j]            # instance in training data with 30 features
            differences = x_train - x_test  # list of differences between training set and test set for all features
            e_distance = np.sqrt(sum(differences)**2) # the euclidean distance between the test vector and training vector
            euclidean_distances.append(e_distance) 
            
        if euclidean_matrix is not None: # if the euclidean matrix already exists
            matrix_new = np.vstack((euclidean_matrix, euclidean_distances)) # add the euclidean distances to the matrix
            euclidean_matrix = matrix_new
        else: # if it does not exist yet:
            euclidean_matrix = np.array([euclidean_distances]) # create the euclidean distance matrix
            
    return euclidean_matrix 

def kNN_prediction_classification(euclidean_matrix, y_train, y_test, k, good_pred_list):
    """
    Function that classifies the k-Nearest neighbors 
    param euclidean_matrix: matrix, with distances between matrices with test en training data
    param y_train: vector, with targets from the training set of the data
    param y_test: vector, with targets from the test set of the data
    param k: integer, represents the number of neighbors 
    param good_pred_list: list, containing the rate of good predictions compared to all predictions per k
    return good_pred_list: updated list, containing the rate of good predictions compared to all predictions, new k added
    """
    
    good_pred = 0
    
    for i in range(len(euclidean_matrix)): # iterate over the rows in the matix, each row consisting of all distances 
        row = euclidean_matrix[i]
        indices_k_closest = sorted(range(len(row)), key = lambda sub: row[sub])[:k] # list of the indices of the K closest vectors
        targets_list = []   #list for the targets of these closest vectors
        
        for index in indices_k_closest:
            targets_list.append(y_train[index])  
        y_pred = max(set(targets_list), key = targets_list.count) # the predicted target is the most common target among the closest vectors
        y_true = y_test[i]    # the true value of the target
        
        if y_true == y_pred:  # if these targets are the same, continue
            good_pred += 1    # count one good prediction
            
    rate_good_pred = good_pred/len(euclidean_matrix) # rate of good predictions compared to all predictions
    good_pred_list.append(rate_good_pred)

    return good_pred_list

def kNN_prediction_regression(euclidean_matrix, y_train, y_test, K):
    """
    Function that computes a k-NN regression
    param euclidean_matrix: matrix, with distances between matrices with test en training data
    param y_train: vector, with targets from the training set of the data
    param y_test: vector, with targets from the test set of the data
    param k: integer, represents the number of neighbors 
    return y_pred_list: list, with values of y predicted by the kNN model
    """
    
    y_pred_list=[]
    
    for i in range(len(euclidean_matrix)): # iterate over the rows in the matix, each row consisting of all distances 
        row = euclidean_matrix[i]
        indices_k_closest = sorted(range(len(row)), key = lambda sub: row[sub])[:K] # list of the indices of the k closest vectors
        targets_list = [] # list for the targets of these closest vectors
        
        for index in indices_k_closest:
            targets_list.append(y_train[index])
            
        y_pred = sum(targets_list)/len(targets_list) # the predicted target is the mean target among the closest vectors
        y_pred_list.append(y_pred)
        
    return y_pred_list        