#Code for the k-NN classification and k-NN regression
#Group: 4, Course: 8DM50

import numpy as np

def eucledian_distance(X_test, X_train):
    euclidean_matrix =None
    for i in range(len(X_test)):        
        x_test = X_test[i]              # one instance
        euclidean_distances = []        # list for distances of instance to training data instances
        for j in range(len(X_train)):   # Loop over the instances in the training data
            x_train = X_train[j]        # instance in training data with 30 features
            differences = x_train - x_test # List of differences between training set and test set for all features
            e_distance = np.sqrt(sum(differences)**2) # The euclidean distance between the test vector and training vector
            euclidean_distances.append(e_distance) 
        if euclidean_matrix is not None: # If the euclidean matrix already exists
            matrix_new = np.vstack((euclidean_matrix, euclidean_distances)) # add the euclidean distances to the matrix
            euclidean_matrix = matrix_new
        else:     # If it does not exist yet:
            euclidean_matrix = np.array([euclidean_distances]) # create the euclidean distance matrix
    return euclidean_matrix 

def kNN_prediction_classification(euclidean_matrix, y_train, y_test, K, good_pred_list):
    good_pred = 0
    for i in range(len(euclidean_matrix)): # Iterate over the rows in the matix, each row consisting of all distances 
        row = euclidean_matrix[i]
        indices_k_closest = sorted(range(len(row)), key = lambda sub: row[sub])[:K] # List of the indices of the K closest vectors
        targets_list = [] # List for the targets of these closest vectors
        for index in indices_k_closest:
            targets_list.append(y_train[index])  
        y_pred = max(set(targets_list), key = targets_list.count) # The predicted target is the most common target among the closest vectors
        y_true = y_test[i]    # The true value of the target
        if y_true == y_pred:  # If these targets are the same:
            good_pred += 1    # Count one good prediction
    rate_good_pred = good_pred/len(euclidean_matrix) # Rate of good predictions compared to all predictions
    good_pred_list.append(rate_good_pred)
    print('Calculation done for K =', K)
    return good_pred_list

def kNN_prediction_regression(euclidean_matrix, y_train, y_test, K):
    y_pred_list=[]
    for i in range(len(euclidean_matrix)): # Iterate over the rows in the matix, each row consisting of all distances 
        row = euclidean_matrix[i]
        indices_k_closest = sorted(range(len(row)), key = lambda sub: row[sub])[:K] # List of the indices of the K closest vectors
        targets_list = [] # List for the targets of these closest vectors
        for index in indices_k_closest:
            targets_list.append(y_train[index])   
        y_pred=sum(targets_list)/len(targets_list)# The predicted target is the mean target among the closest vectors
        y_pred_list.append(y_pred)
    return y_pred_list
        