#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from collections import OrderedDict
import time
import matplotlib.pyplot
import scipy as sp


# In[2]:


#Import the test data into a single pandas dataframe with labels as a column
test_data = pd.read_csv("faces/X_test.csv", header=None)
test_labels = pd.read_csv("faces/Y_test.csv", names=['label'])
test_data = test_data.join(test_labels)
test_data


# In[3]:


#Import the training data into a single pandas dataframe with labels as a column
train_data = pd.read_csv("faces/X_train.csv", header=None)
train_labels = pd.read_csv("faces/Y_train.csv", names=['label'])
train_data = train_data.join(train_labels)
train_data


# In[4]:


#display a photo
matplotlib.pyplot.imshow(np.reshape(np.array(train_data.iloc[10,:-1]),(92,112)),cmap="gray")


# # (a) Visualizing Eigenfaces

# In[5]:


#Subtract the mean from the columns and save as new dataframe
norm = train_data.iloc[:,:-1].sub(train_data.iloc[:,:-1].mean(0), axis=1)

#Create covariance matrix
cov = norm.transpose().dot(norm)

#Calculate the eigenvetors from 50 largest eigenvalues
e_val, e_vec = sp.sparse.linalg.eigs(cov.values, k=50)

#Keep only the real portion of the eigen-things
e_val = np.real(e_val)
e_vec = np.real(e_vec)

#Create a new dataframe with index of eigenvalues and rows of eigenvectors
eig = pd.DataFrame(e_vec, columns = e_val).transpose()


# In[6]:


eig


# In[7]:


#Plot and save the top 10 eigenvalues associated vectors as images. Creepy....
for i in range (10):
    matplotlib.pyplot.imshow(np.reshape(np.array(eig.iloc[i]),(92,112)),cmap="gray")
    
    #.figure.savefig("Q3_eig_"+str(i+1)+".pdf")


# # (b) Face Reconstruction

# In[8]:


#Calculate the projections of normalized images [1,5,20,30,40] and then re-map them to the vector space and show them
u=[]
x_prime=[]
projections = {}
for i, x in norm.iloc[[0,4,19,29,39]].iterrows():
    u.append(eig.apply(lambda V: x.transpose().dot(V), axis =1))
    x_prime.append(u[-1].dot(eig).values)
    matplotlib.pyplot.imshow(np.reshape(x.values,(92,112)),cmap="gray")
    #.figure.savefig("Q3_b_r"+str(i+1)+".pdf")
    matplotlib.pyplot.imshow(np.reshape(x_prime[-1],(92,112)),cmap="gray")
    #.figure.savefig("Q3_b_f"+str(i+1)+".pdf")
    projections[i] = u[-1]


# # (c) Face Recognition

# In[9]:


#Subtract the mean from the training data from the test data
n = test_data.iloc[:,:-1].sub(train_data.iloc[:,:-1].mean(0), axis=1)

#Calculate the projections of normalized training and test images in order to reduce dimensionality 

u=[]
x_prime=[]
projections_train = {}
for i, x in norm.iterrows():
    u.append(eig.apply(lambda V: x.transpose().dot(V), axis =1))
    projections_train[i] = u[-1]

u=[]
x_prime=[]
projections_test = {}
for i, x in n.iterrows():
    u.append(eig.apply(lambda V: x.transpose().dot(V), axis =1))
    projections_test[i] = u[-1]
    
#Asign dimension-reduced data to dataframes
X_test = pd.DataFrame.from_dict(projections_test).transpose()
X_train = pd.DataFrame.from_dict(projections_train).transpose()


# In[10]:


#Train and test a KNN where k=1
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, train_labels.values)
clf.score(X_test, test_labels.values)

