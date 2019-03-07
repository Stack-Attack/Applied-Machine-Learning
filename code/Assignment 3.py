#!/usr/bin/env python
# coding: utf-8

# In[82]:


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


# In[83]:


#Stores filenames for import
files = []

#Creates lists of filesnames for import
for i in range(1000):
    files.append('review_polarity/txt_sentoken/pos/t ('+str(i+1)+').txt')
    files.append('review_polarity/txt_sentoken/neg/f ('+str(i+1)+').txt')

#Load data into a data frame using a CountVectorizer
vectorizor = CountVectorizer(input='filename', stop_words='english')
data = vectorizor.fit_transform(files)
data = pd.DataFrame(data.todense(), columns=vectorizor.get_feature_names())

#Assign labels to dataframe based on import order
data['label'] = ['pos']*1000 + ['neg']*1000

#Shuffle the data
data = shuffle(data)

#Split data
test_data = data.tail(400).reset_index(drop=True)
T = data.iloc[0:100].reset_index(drop=True)
T_r = data.tail(1500).reset_index(drop=True)


# # (i)

# ## Uncertainty Sampling

# In[86]:


#Create linear SVM classifier
svc = LinearSVC(max_iter=10000)

#Constant that changes the number of labeled data added each loop
K = 10

#Copy the data as it will be mutated
q1_T = T.copy()
q1_T_r = T_r.copy()

#Used to store results. Keys will be the number of training samples
q1_results = OrderedDict()

#Initialize first key
key=100

#Loop untill break
while True:
    start = time.time()
    
    #Fit the model, and store results
    svc.fit(q1_T.drop(['label'],axis=1),q1_T['label'])
    q1_results[key] = [svc.score(test_data.drop(['label'],axis=1),test_data['label'])]
    
    #If there is data left to look through then use it
    if (len(q1_T_r)>0):
        
        #Generate uncertainty metric and take it's absolute value (approximates distance from margin)
        q1_T_r['un'] = np.absolute(svc.decision_function(q1_T_r.drop(['label'],axis=1)))
        
        #Sort the uncertainty to find the points closest to margin
        q1_T_r = q1_T_r.sort_values(by='un')
        
        #Add the K points with highest uncertanty to the training data
        q1_T = pd.concat([q1_T,q1_T_r.head(K).drop(['un'],axis=1)], ignore_index = True)
        
        #Drop the K points from the remainder set
        q1_T_r = q1_T_r.iloc[K:].drop(['un'],axis=1)
        end = time.time()
        print("Samples:"+str(key))
        print(end - start)
        
        #Increment key
        key+=K
        
    else: break
    
q1_results


# ## Random Sampling

# In[89]:


#This section repeats the previous methods however it picks random points to label
svc = LinearSVC(max_iter=10000)


K = 10

random_q1_T = T.copy()
random_q1_T_r = T_r.copy()

random_q1_results = OrderedDict()
key=100
while True:
    start = time.time()
    svc.fit(random_q1_T.drop(['label'],axis=1),random_q1_T['label'])
    random_q1_results[key] = [svc.score(test_data.drop(['label'],axis=1),test_data['label'])]
    
    if (len(random_q1_T_r)>0):
        random_q1_T = pd.concat([random_q1_T,random_q1_T_r.head(K)], ignore_index = True)
        random_q1_T_r = random_q1_T_r.iloc[K:]
        end = time.time()
        print("Samples:"+str(key))
        print(end - start)
    
        key+=K
        
    else: break
    
random_q1_results


# ## Results

# In[106]:


#Create and save plots
q1_results_test = pd.concat([pd.DataFrame.from_dict(q1_results), pd.DataFrame.from_dict(random_q1_results)]).transpose()
q1_results_test.columns = ["Uncertainty Sampling","Random Sampling"]

plot = q1_results_test.plot.line(title='Linear SVM Classification Performance Using Uncertainty Sampling')
plot.set_xlabel("Size of Training Dataset")
plot.set_ylabel("Average Test Accuracy")

#Display plot in Jupyter
plot

#Save plot
fig = plot.get_figure()
fig.savefig("Q2_i.pdf")


# # (ii)

# ## Query by Committee

# In[107]:


#Initialize three kinds of classifiers
svc = LinearSVC(max_iter=10000) #Support Vector Machine
dtc = DecisionTreeClassifier() #Decision Tree
knc = KNeighborsClassifier(n_jobs=-1) #K-Nearest Neighbors

#K-value 
K = 10

#Copy the data as it will be mutated
q2_T = T.copy()
q2_T_r = T_r.copy()
test_data = test_data.copy()

#Used to store results where key is the number of training data
q2_results = OrderedDict()
key=100

#Repeat untill break (no more data)
while True:
    #Fit all three models using initial points
    svc.fit(q2_T.drop(['label'],axis=1),q2_T['label'])
    dtc.fit(q2_T.drop(['label'],axis=1),q2_T['label'])
    knc.fit(q2_T.drop(['label'],axis=1),q2_T['label'])

    
    start = time.time()

    #Store the test predictions in a dictionary
    test_predictions = {
        #Predictions are mapped to 1 or -1 for positive and negative respectively for easy voting
        'svc': list(map(lambda x: 1 if x=='pos' else -1 ,svc.predict(test_data.drop(['label'],axis=1)))),
        'dtc': list(map(lambda x: 1 if x=='pos' else -1 ,dtc.predict(test_data.drop(['label'],axis=1)))),
        'knc': list(map(lambda x: 1 if x=='pos' else -1 ,knc.predict(test_data.drop(['label'],axis=1))))
    }
    
    #If the sum of a data points 3 predictions is >= 1 then majority voted positive, otherwise negative
    test_predictions = list(map(lambda a, b, c: 'pos' if (a+b+c)>=1 else 'neg', test_predictions['svc'], test_predictions['dtc'], test_predictions['knc']))
    
    #This could be turned into a zipped filter
    #Caclulates the number of corect predictions and devides by total for accuracy
    c=0
    for i, j in zip(test_data['label'], test_predictions):
        if i==j:
            c+=1
    q2_results[key]=[c/len(test_data['label'])]

    #If there is data left to look through then use it
    if (len(q2_T_r)>0):
        #Predictions for remaining data are mapped to 1 or -1 for positive and negative respectively for easy voting
        train_predictions = {
            'svc': list(map(lambda x: 1 if x=='pos' else -1 ,svc.predict(q2_T_r.drop(['label'],axis=1)))),
            'dtc': list(map(lambda x: 1 if x=='pos' else -1 ,dtc.predict(q2_T_r.drop(['label'],axis=1)))),
            'knc': list(map(lambda x: 1 if x=='pos' else -1 ,knc.predict(q2_T_r.drop(['label'],axis=1))))
        }
        
        #If the min sum of a data points 3 predictions is 1. Abs(sum) is used to classify as disagreeable or agreeable
        q2_T_r['un'] = list(map(lambda a, b, c: 'disagree' if abs(a+b+c)<=1 else 'agree', train_predictions['svc'], train_predictions['dtc'], train_predictions['knc']))
        
        #Sort the values so disagreeable ones are on top
        q2_T_r = q2_T_r.sort_values(by='un', ascending=False)
        
        #Add 10 disagreeable point to the training set
        q2_T = pd.concat([q2_T,q2_T_r.head(K).drop(['un'],axis=1)], ignore_index = True)
        
        #Remove those points from the remainder set
        q2_T_r = q2_T_r.iloc[K:].drop(['un'],axis=1)

        end = time.time()
        print("Samples:"+str(key))
        print(q2_results[key])
        print(end - start)

        key+=K

    else: break

        
q2_results


# ## Random Sampling

# In[110]:


#Same as above calculations except new training data is picked randomly

svc = LinearSVC(max_iter=10000)
dtc = DecisionTreeClassifier()
knc = KNeighborsClassifier(n_jobs=-1)


K = 10

random_q2_T = T.copy()
random_q2_T_r = T_r.copy()
test_data = test_data.copy()

random_q2_results = OrderedDict()
key=100
while True:
    svc.fit(random_q2_T.drop(['label'],axis=1),random_q2_T['label'])
    dtc.fit(random_q2_T.drop(['label'],axis=1),random_q2_T['label'])
    knc.fit(random_q2_T.drop(['label'],axis=1),random_q2_T['label'])

    
    start = time.time()

    #Store the test predictions in a dictionary
    random_test_predictions = {
        #Predictions are mapped to 1 or -1 for positive and negative respectively for easy voting
        'svc': list(map(lambda x: 1 if x=='pos' else -1 ,svc.predict(test_data.drop(['label'],axis=1)))),
        'dtc': list(map(lambda x: 1 if x=='pos' else -1 ,dtc.predict(test_data.drop(['label'],axis=1)))),
        'knc': list(map(lambda x: 1 if x=='pos' else -1 ,knc.predict(test_data.drop(['label'],axis=1))))
    }
    
    #If the sum of a data points 3 predictions is >= 1 then majority voted positive, otherwise negative
    random_test_predictions = list(map(lambda a, b, c: 'pos' if (a+b+c)>=1 else 'neg', random_test_predictions['svc'], random_test_predictions['dtc'], random_test_predictions['knc']))
    
    #This could be turned into a zipped filter
    #Caclulates the number of corect predictions and devides by total for accuracy
    c=0
    for i, j in zip(test_data['label'], random_test_predictions):
        if i==j:
            c+=1
    random_q2_results[key]=[c/len(test_data['label'])]
    

    if (len(random_q2_T_r)>0):
        random_q2_T = pd.concat([random_q2_T,random_q2_T_r.head(K)], ignore_index = True)
        random_q2_T_r = random_q2_T_r.iloc[K:]
        
        end = time.time()
        print("Samples:"+str(key))
        print(random_q2_results[key])
        print(end - start)

        key+=K

    else: break

        
random_q2_results


# ## Results

# In[119]:


#Create plots and save them

q2_results_test = pd.concat([pd.DataFrame.from_dict(q2_results), pd.DataFrame.from_dict(random_q2_results)]).transpose()
q2_results_test.columns = ["QbC Sampling","Random Sampling"]

plot = q2_results_test.plot.line(title='Linear SVM Classification Performance Using Query by Committee')
plot.set_xlabel("Size of Training Dataset")
plot.set_ylabel("Average Test Accuracy")

#Display plot in Jupyter
plot

#Save plot
fig = plot.get_figure()
fig.savefig("Q2_ii.pdf")


# In[ ]:




