
# coding: utf-8

# In[99]:


import numpy as np
import pandas as pd
import scipy as sp
import math
from random import randint


# In[100]:


#data_folds gathers all of the data fro, the seperate files for concatination
data_folds = []

#Each data and label file is combined into one dataframe per fold
#The labels are added as a new column titled 'label'
#The fold is added as a new column titled 'fold'
for i in range(10):
    labels = pd.read_csv("knn-dataset/labels"+str((i+1))+".csv", names=['label'])
    data_folds.append(pd.read_csv("knn-dataset/data"+str((i+1))+".csv", header=None))
    data_folds[i] = data_folds[i].join(labels)
    data_folds[i]['fold']=i+1

#The individual data_folds dataframes are combined into one called 'data'
data = pd.concat(data_folds, ignore_index = True)
data


# In[115]:


# https://en.wikipedia.org/wiki/Euclidean_distance


def knn_predict(test_data, point, k, n=64):
    """Tests a single multi-dimensional data point against labeled data. The test
    dataframe should contain a column with actual labels named 'label'
    
    test_data: pandas dataframe containing n-columns of numerical attributes
    point: pandas series containing n-columns of numerical attributes
    k: number of neighbors to test against
    n: number of attributes (must be the first columns in dataframe)
    
    Returns a pandas Series of classifications for each value <=k"""
    
    #Reset the data index in case data is from the same dataframe
    test_data.reset_index(drop=True,inplace=True)
    
    #Subtract and aquare the point's attributes from the test_data attributes
    #Replaces the test_data attribute values with the results
    for column in test_data.iloc[:,:n]:
        test_data[column] = (point[column]-test_data[column])**2
    
    #Apply a sum function row-wise and square the result
    #Create a new column called 'dist' where euclidean distance is stored
    test_data['dist'] = test_data.apply(lambda row: math.sqrt(np.sum(row)), axis=1)
    
    #Sort the dataframe by distance (lowest first)
    test_data = test_data.sort_values(by=['dist'])
    
    #For each value <=k calculate the mode of the labels
    #Store the mode for each assosiated k in an array titled'labels'
    #In the event of a tie select the mode randomly
    labels = []
    for i in range(k):
        mode = test_data.head(i+1)['label'].mode()
        if mode.size > 1:
            labels.append(mode.iloc[randint(0,1)])
        else:
            labels.append(mode.iloc[0])
            
    #Return a pandas Series of classifications for each value <=k
    return pd.Series(labels)


def knn(train, test, k=30, n=64):
    """Tests a data set using KNN under given parameters.
    
    train: pandas dataframe with labels in a 'label' column and n attribute columns
    test: pandas dataframe with n-attribute columns (must be first)
    k: number of neighbors to test against
    n: number of attributes (must be the first columns in dataframe)
    
    Returns a pandas dataframe of classifications for values <=k (maintains label columns)"""
    
    #For all values <=k create a new column named 'k_#' where # is the k-value
    column_names = []
    for i in range(k):
        column_names.append('k_'+str(i+1))
        
    #Apply the knn_predict function to each point in the test dataframe
    #Store results in the associated 'k_#' column for each point in the test dataframe
    test[column_names] = test.apply(lambda row: knn_predict(train.copy(), row, k, n), axis=1)
    column_names.extend(['label'])
    
    #Return dataframe of 'k_#' results, and label columns
    return test[column_names]


def knn_10_fold(data, k=30, n=64):
    """Compute the accuracy of KNN accross difference values <=k (# of neighbors not folds)
    
    data: pandas dataframe with labels in a 'label' column, folds in the 'fold column and n attribute columns
    k: number of neighbors to test against
    n: number of attributes (must be the first columns in dataframe)
    
    Returns pandas series of associated accuracies for values <=k"""
    
    #The knn function is run by finding each fold and testing against all others
    #The results of each folds testing are stored in the 'folds' array as pandas dataframes
    folds = []
    for i in range(10):
        folds.append(knn(data.loc[data['fold']!=i+1].copy(), data.loc[data['fold']==i+1].copy(),k))
    
    #Combine knn results from all folds into one dataframe
    data = pd.concat(folds)
    
    def verify_labels(row):
        """Given a row return a row with 1's and 0's after comparing against label
        
        row: pandas series of calssifications for values <=k
        
        Return pandas series with binary values for correct/incorrect classifications"""
        
        #Store the correct label for each row
        correct_label = row.iloc[k]
        
        #Modify each column to a 1 or 0 based on comparison against correct_label
        for i in range(k):
            if row.iloc[i] == correct_label:
                row.iloc[i] = 1
            else:
                row.iloc[i] = 0
        
        #Return a pandas series of classification results for values <=k
        return row
    
    #Apply the verify labels function row-wise
    data.apply(verify_labels, axis=1)
    
    #Drop the label column as it's no longer needed
    data = data.drop(['label'], axis=1)
    
    def get_means(column):
        """Get the means of all columns.
        
        column: pandas series
        
        Returns the mean of the pandas series"""
        return column.mean()
    
    #Apply the get_means function column-wise and store the results (pandas series) in 'performance'
    performance = data.apply(get_means)
    
    #Plot the results
    performance.plot.line()
    
    #Return the pandas series containing performance results
    return performance

#Run the 10-fold KNN validation on the data
results = knn_10_fold(data)

print(results)

