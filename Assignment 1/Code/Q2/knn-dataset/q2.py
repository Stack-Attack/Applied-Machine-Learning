
# coding: utf-8

# In[99]:


import numpy as np
import pandas as pd
import scipy as sp
import math
from random import randint


# In[100]:


data_folds = []

for i in range(10):
    labels = pd.read_csv("knn-dataset/labels"+str((i+1))+".csv", names=['label'])
    data_folds.append(pd.read_csv("knn-dataset/data"+str((i+1))+".csv", header=None))
    data_folds[i] = data_folds[i].join(labels)
    data_folds[i]['fold']=i+1
    
data = pd.concat(data_folds, ignore_index = True)
data


# In[ ]:


# https://en.wikipedia.org/wiki/Euclidean_distance

def knn_predict(test_data, point, k):
    #Reset the data index as a point was removed
    test_data.reset_index(drop=True,inplace=True)
    
    for column in test_data.iloc[:,:64]:
        test_data[column] = (point[column]-test_data[column])**2
        
    test_data['dist'] = test_data.apply(lambda row: math.sqrt(np.sum(row)), axis=1)
    test_data = test_data.sort_values(by=['dist'])
    
    labels = []
    
    for i in range(k):
        mode = test_data.head(i+1)['label'].mode()
        if mode.size > 1:
            labels.append(mode.iloc[randint(0,1)])
        else:
            labels.append(mode.iloc[0])
    return pd.Series(labels)

def knn(train, test, k=30):
    column_names = []
    for i in range(k):
        column_names.append('k_'+str(i+1))
    test[column_names] = test.apply(lambda row: knn_predict(train.copy(), row, k), axis=1)
    column_names.extend(['label', 'fold'])
    return test[column_names]


#knn(data.iloc[100:].copy(), data.head(100).copy())


def knn_10_fold(data, k=30):
    folds = []
    
    for i in range(10):
        folds.append(knn(data.loc[data['fold']!=i+1].copy(), data.loc[data['fold']==i+1].copy()).drop(['fold'], axis=1))
    
    data = pd.concat(folds)
    
    def verify_labels(row):
        correct_label = row.iloc[k]
        for i in range(k):
            if row.iloc[i] == correct_label:
                row.iloc[i] = 1
            else:
                row.iloc[i] = 0
        return row
    
    data.apply(verify_labels, axis=1)
    data = data.drop(['label'], axis=1)
    
    def get_means(column):
        return column.mean()
    
    performance = data.apply(get_means)
    performance.plot.line()
    
    return data
    
folds = knn_10_fold(data)


# In[87]:


k = 30

def verify_labels(row):
    correct_label = row.iloc[k]
    for i in range(k):
        if row.iloc[i] == correct_label:
            row.iloc[i] = 1
        else:
            row.iloc[i] = 0
    return row
folds[1].apply(verify_labels, axis=1)


# In[106]:


folds

