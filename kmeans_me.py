# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read the dataset
dataset = pd.read_csv('C:/Users/Nikitha/.spyder-py3/K_Means/Mall_Customers.csv')
X = dataset.iloc[:,3:5].values

#Find the number of clusters to segment the customers

from sklearn.cluster import KMeans

#Elbow method
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('Elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()  

#Fit the clusters to Dataset
kmeans = KMeans(n_clusters = 5, max_iter = 300, n_init =10 , random_state = 0)
y_pred = kmeans.fit_predict(X)

#Visualizing the clusters
plt.scatter(X[y_pred==0,0], X[y_pred==0,1], s=100, c = 'red', label = 'Careful')
plt.scatter(X[y_pred==1,0], X[y_pred==1,1], s=100, c = 'blue', label = 'Standard')
plt.scatter(X[y_pred==2,0], X[y_pred==2,1], s=100, c = 'green', label = 'Target')
plt.scatter(X[y_pred==3,0], X[y_pred==3,1], s=100, c = 'pink', label = 'Careless')
plt.scatter(X[y_pred==4,0], X[y_pred==4,1], s=100, c = 'orange', label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=200, c ='yellow', label='Centroid')
plt.title('Cluster of Customers')
plt.xlabel('Annual Income$')
plt.ylabel('Spending score(1-100)')
plt.legend()
plt.show()







