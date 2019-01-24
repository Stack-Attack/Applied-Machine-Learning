
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy as sp
import math
from sklearn import tree, neighbors
from sklearn.utils import shuffle
from random import randint
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from collections import OrderedDict


# In[2]:


#Headers sets names for all of the attributes contained in the data set
headers = ["id",
           "label",
           "mean_radius",
           "mean_texture",
           "mean_perimeter",
           "mean_area",
           "mean_smoothness",
           "mean_compactness",
           "mean_cancavity",
           "mean_concave_points",
           "mean_symmetry",
           "mean_fractal_dimension",
           "sd_radius",
           "sd_texture",
           "sd_perimeter",
           "sd_area",
           "sd_smoothness",
           "sd_compactness",
           "sd_cancavity",
           "sd_concave_points",
           "sd_symmetry",
           "sd_fractal_dimension",
           "max_radius",
           "max_texture",
           "max_perimeter",
           "max_area",
           "max_smoothness",
           "max_compactness",
           "max_cancavity",
           "max_concave_points",
           "max_symmetry",
           "max_fractal_dimension"]
                
#Data is read into test/train pandas dataframes while being shuffled using a seed of 4 (because I like 4)
test_data = shuffle(pd.read_csv("wdbc-test.csv", names = headers, index_col=0), random_state=4)
train_data = shuffle(pd.read_csv("wdbc-train.csv", names = headers, index_col=0), random_state=4)



# Not needed anymore! Thanks sklearn!
# 
# def binarize(val):
#     if val == 'B':
#         return 0
#     else:
#         return 1
# 
# test_data['label'] = test_data['label'].map(binarize)
# train_data['label'] = train_data['label'].map(binarize)

# In[3]:


test_data


# In[4]:


train_data


# In[5]:


#list of the names of the attributes to be used for classification
attributes = ["mean_radius",
              "mean_texture",
              "mean_perimeter",
              "mean_area",
              "mean_smoothness",
              "mean_compactness",
              "mean_cancavity",
              "mean_concave_points",
              "mean_symmetry",
              "mean_fractal_dimension",
              "sd_radius",
              "sd_texture",
              "sd_perimeter",
              "sd_area",
              "sd_smoothness",
              "sd_compactness",
              "sd_cancavity",
              "sd_concave_points",
              "sd_symmetry",
              "sd_fractal_dimension",
              "max_radius",
              "max_texture",
              "max_perimeter",
              "max_area",
              "max_smoothness",
              "max_compactness",
              "max_cancavity",
              "max_concave_points",
              "max_symmetry",
              "max_fractal_dimension"]


# # A)
# 
# ## Decision Tree

# In[6]:


#Initialize a decision tree classifier with a random seed of 4 for repeatability
dt_clf = tree.DecisionTreeClassifier(random_state=4)

#Initialize a python dictonary containing the hyperparameters to test and their ranges (depth from 1-100)
param_grid = {'max_depth': np.arange(1, 100)}

#initialize and test the hyperparameters using 10-fold validation on the training data
gs_dt_clf = GridSearchCV(dt_clf, param_grid, cv=10)
gs_dt_clf.fit(train_data.loc[:,attributes], train_data.label)
print(gs_dt_clf.best_params_)

#Re-initialize a decision tree with the best max_depth parameter
dt_clf = tree.DecisionTreeClassifier(max_depth=gs_dt_clf.best_params_['max_depth'], random_state=4)

#Fit the decision tree using the training data and the training labels
dt_clf = dt_clf.fit(train_data.loc[:,attributes], train_data.label)


# In[7]:


#Display a visual representation of the tree
#https://scikit-learn.org/stable/modules/tree.html#classification
#import graphviz
#dot_data = tree.export_graphviz(dt_clf, out_file=None, 
#                                feature_names=attributes,  
#                                class_names=["B","M"],  
#                                filled=True, rounded=True,  
#                                special_characters=True,
#                                ) 
#graph = graphviz.Source(dot_data) 
#graph.render('Q4_tree')
#graph


# ## KNN

# In[8]:


#Initialize a KNN classifier
knn_clf = neighbors.KNeighborsClassifier()

#Dictionary used to store accuracies of associated k-values. Maintains order 
k_results = OrderedDict()

#Stores the best score and the best k through the loop
best = 0
best_k = 0

#Initialize a KFold object for splitting data
kf = KFold(n_splits=10)

#Look through 100 k values
for k in range(100):
    
    #Store the resulting accuracies of each k
    results = np.array([])
    
    #Work your way through the indices of each fold using the KFold object
    for train_index, test_index in kf.split(train_data.loc[:,attributes], y=train_data['label']):
        #Initialize train and test attribute values by splitting data on provided indices
        X_train, X_test = train_data.loc[:,attributes].iloc[train_index], train_data.loc[:,attributes].iloc[test_index]
        #Initialize train and test label values by splitting data on provided indices
        y_train, y_test = train_data['label'].iloc[train_index], train_data['label'].iloc[test_index]
        
        #Set the number of neighbors in the algorithm to k+1 (starts at 0)
        knn_clf.set_params(n_neighbors=k+1)
        #Fit the KNN model using the training data
        knn_clf = knn_clf.fit(X_train, y_train)
        #Append the results using from the test data to the results array
        results = np.append(results, [knn_clf.score(X_test, y_test)])
        
    #Apply the accuracy of k accross all folds to the dictionary in-order
    k_results["k_"+str(k+1)] = np.average(results)
    
    #If the accuracy of this k was better than the best replace the best track it
    if np.average(results) > best :
        best = np.average(results)
        best_k = k
        
#Plot the results
pd.Series(k_results).plot.line()
print("Best k: " + str(best_k))

#Re-initialize the knn using the best k
knn_clf = neighbors.KNeighborsClassifier(n_neighbors=best_k)



# # B)

# ## Decision Tree

# In[9]:


#Get the training score and predicted values using 10-fold cross validation
dt_train_score = cross_val_score(dt_clf, train_data.loc[:,attributes], train_data["label"], cv=10)
dt_train_pred = cross_val_predict(dt_clf, train_data.loc[:,attributes], train_data["label"], cv=10)

#Create a confusion matrix and print other performance metrics using computed values
dt_train_conf = confusion_matrix(train_data["label"], dt_train_pred)
print(classification_report(train_data["label"], dt_train_pred))

print(np.mean(dt_train_score))
print(dt_train_conf)


# ## KNN

# In[10]:


#Get the training score and predicted values using 10-fold cross validation
knn_train_score = cross_val_score(knn_clf, train_data.loc[:,attributes], train_data["label"], cv=10)
knn_train_pred = cross_val_predict(knn_clf, train_data.loc[:,attributes], train_data["label"], cv=10)

#Create a confusion matrix and print other performance metrics using computed values
knn_train_conf = confusion_matrix(train_data["label"], knn_train_pred)
print(classification_report(train_data["label"], dt_train_pred))


print(np.mean(knn_train_score))
print(knn_train_conf)


# # C)

# ## Decision Tree

# In[11]:


#Get the training score and predicted values using 10-fold cross validation
dt_clf = dt_clf.fit(train_data.loc[:,attributes], train_data.label)
dt_test_pred = dt_clf.predict(test_data.loc[:,attributes])

#Create a confusion matrix and print other performance metrics using computed values
dt_test_conf = confusion_matrix(test_data["label"], dt_test_pred)
print(classification_report(test_data["label"], dt_test_pred))

print(dt_test_conf)


# ## KNN

# In[12]:


#Get the training score and predicted values using 10-fold cross validation
knn_clf = knn_clf.fit(train_data.loc[:,attributes], train_data['label'])
knn_test_pred = knn_clf.predict(test_data.loc[:,attributes])

#Create a confusion matrix and print other performance metrics using computed values
knn_test_conf = confusion_matrix(test_data["label"], knn_test_pred)
print(classification_report(test_data["label"], knn_test_pred))

print(knn_test_conf)

