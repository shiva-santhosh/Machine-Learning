#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from imageio import imread


# In[27]:


#Task-1

X_1 = np.random.multivariate_normal(mean=[4, 0], cov=[[1, 0], [0, 1]], size=75)
X_2 = np.random.multivariate_normal(mean=[6, 6], cov=[[2, 0], [0, 2]], size=250)
X_3 = np.random.multivariate_normal(mean=[1, 5], cov=[[1, 0], [0, 2]], size=20)


# In[36]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Dimension-1', fontsize = 15)
ax.set_ylabel('Dimension-2', fontsize = 15)
ax.set_title('(Initial Plot)Three Clusters', fontsize = 20)
ax.scatter(X_1[:,0],X_1[:,1],label='X_1 ponts')
ax.scatter(X_2[:,0],X_2[:,1],label='X_2 points')
ax.scatter(X_3[:,0],X_3[:,1],label='X_3 points')
leg = ax.legend()


# In[29]:


#task-2 

training_data = list(X_1[:]) + list(X_2[:]) + list(X_3[:])


# In[30]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(training_data)
labels = kmeans.predict(training_data)
centroids = kmeans.cluster_centers_
print("The centriods found for three different clusters are")
print(centroids)


# In[31]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Dimension-1', fontsize = 15)
ax.set_ylabel('Dimension-2', fontsize = 15)
ax.set_title('(After training)K-Means Clustering', fontsize = 20)
ax.scatter(X_1[:,0],X_1[:,1],label='X_1 ponts')
ax.scatter(X_2[:,0],X_2[:,1],label='X_2 points')
ax.scatter(X_3[:,0],X_3[:,1],label='X_3 points')
ax.scatter(centroids[:,0],centroids[:,1],s=200,color="black",label='Centriods',marker='*')
leg=ax.legend()
print("X_1 points are labelled with blue colour")
print("X_2 points are labelled with orange colour")
print("X_3 points are labelled with green colour")
print("Black points are the found centriods for the 3 clusters using k-means clustering")


# In[32]:


print("Repeating the same thing for different center")
print("The random state is initilaised to be random and at every time it will pick a random point")
kmeans = KMeans(n_clusters=3)
kmeans.fit(training_data)
labels = kmeans.predict(training_data)
centroids = kmeans.cluster_centers_
print("The centriods found for three different clusters are")
print(centroids)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Dimension-1', fontsize = 15)
ax.set_ylabel('Dimension-2', fontsize = 15)
ax.set_title('(2nd Second time)K-Means Clustering', fontsize = 20)
ax.scatter(X_1[:,0],X_1[:,1],label='X_1 ponts')
ax.scatter(X_2[:,0],X_2[:,1],label='X_2 points')
ax.scatter(X_3[:,0],X_3[:,1],label='X_3 points')
ax.scatter(centroids[:,0],centroids[:,1],s=100,color="black",label='Centriods',marker='*')
leg = ax.legend()

print("We can see that initial centriods and this centriods are not exactly same,since the another random center position is been taken ")
print("The centriods value are almost same, since k-means algorithm repeats untill covergence")


# In[37]:


# task-3
gmm = GaussianMixture(n_components = 3) 
gmm.fit(training_data)
labels = gmm.predict(training_data)
print("The centriods of the GMM-model are")
centroids = gmm.means_
print(centroids)


# In[38]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Dimension-1', fontsize = 15)
ax.set_ylabel('Dimension-2', fontsize = 15)
ax.set_title('(After training)GMM', fontsize = 20)
ax.scatter(X_1[:,0],X_1[:,1],label='X_1 ponts')
ax.scatter(X_2[:,0],X_2[:,1],label='X_2 points')
ax.scatter(X_3[:,0],X_3[:,1],label='X_3 points')
ax.scatter(centroids[:,0],centroids[:,1],s=100,color="black",label='Centriods',marker='*')
leg=ax.legend()
print("X_1 points are labelled with blue colour")
print("X_2 points are labelled with orange colour")
print("X_3 points are labelled with green colour")
print("Black points are the found centriods for the 3 clusters using k-means clustering")


# In[35]:


print("Repeating the same thing for different center")
print("The random state is initilaised to be random and at every time it will pick a random point")
gmm = GaussianMixture(n_components = 3) 
gmm.fit(training_data)
labels = gmm.predict(training_data)
centroids = gmm.means_
print(centroids)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Dimension-1', fontsize = 15)
ax.set_ylabel('Dimension-2', fontsize = 15)
ax.set_title('(2nd time)GMM', fontsize = 20)
ax.scatter(X_1[:,0],X_1[:,1],label='X_1 ponts')
ax.scatter(X_2[:,0],X_2[:,1],label='X_2 points')
ax.scatter(X_3[:,0],X_3[:,1],label='X_3 points')
ax.scatter(centroids[:,0],centroids[:,1],s=100,color="black",label='Centriods',marker='*')
leg=ax.legend()
print("X_1 points are labelled with blue colour")
print("X_2 points are labelled with orange colour")
print("X_3 points are labelled with green colour")
print("Black points are the found centriods for the 3 clusters using k-means clustering")


# In[39]:


#task-4 
print("Image compression\n")
img = imread('myphoto.jpg')
img_size = img.shape
print(img_size)
x_2d = img.reshape(img_size[0] * img_size[1], img_size[2])
kmeans = KMeans(n_clusters=30)
kmeans.fit(x_2d)
x_compressed = kmeans.cluster_centers_[kmeans.labels_]
x_compressed = np.clip(x_compressed.astype('uint8'), 0, 255)
x_compressed = x_compressed.reshape(img_size[0], img_size[1], img_size[2])

fig, ax = plt.subplots(1, 2, figsize = (12, 8))
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[1].imshow(x_compressed)
ax[1].set_title('Compressed Image with 30 colors')


# In[1]:





# In[ ]:




