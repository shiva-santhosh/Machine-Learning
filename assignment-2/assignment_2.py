#!/usr/bin/env python
# coding: utf-8

# In[115]:


import numpy as np
from matplotlib import pyplot as plt 
import random
from mpl_toolkits import mplot3d
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[141]:


# when radius is 1 and centered at (1,1,1)
r = 1 ;
x1 = []
y1 = []
z1 = []
o1 = []

for i in range (500) :
    
    phi = random.uniform(0,2* (np.pi))
    costheta = random.uniform(-1,1)
    theta = np.arccos( costheta )
    
    x1.append(1 + r*np.sin(theta)*np.cos(phi))
    y1.append(1 + r*np.sin(theta)*np.sin(phi))
    z1.append(1+r*np.cos(theta)) 
    o1.append('b')

x1 = np.array(x1)
y1 = np.array(y1)
z1 = np.array(z1)

# when radius is 2 and centered at (1,1,1)
r = 2
x2 = []
y2 = []
z2 = []
o2 = []

for i in range (500) :
    
    phi = random.uniform(0,2* (np.pi))
    costheta = random.uniform(-1,1)
    theta = np.arccos( costheta )
    
    x2.append(1 + r*np.sin(theta)*np.cos(phi))
    y2.append(1 + r*np.sin(theta)*np.sin(phi))
    z2.append(1 + r*np.cos(theta)) 
    o2.append('r')

x2 = np.array(x2)
y2 = np.array(y2)
z2 = np.array(z2)

# when radius is 2 and centered at (1,1,1)

r = 3
x3 = []
y3 = []
z3 = []
o3 = []

for i in range (500) :
    
    phi = random.uniform(0,2* (np.pi))
    costheta = random.uniform(-1,1)
    theta = np.arccos( costheta )
    
    x3.append(1 + r*np.sin(theta)*np.cos(phi))
    y3.append(1 + r*np.sin(theta)*np.sin(phi))
    z3.append(1 + r*np.cos(theta))
    o3.append('g')

x3 = np.array(x3)
y3 = np.array(y3)
z3 = np.array(z3)

fig = plt.figure(figsize = (8,8))
ax = plt.axes(projection='3d')

ax.scatter3D(x1, y1, z1, color='blue')
ax.scatter3D(x2, y2, z2, color='red')
ax.scatter3D(x3, y3, z3, color='green')




# # Blue labels indicate sphere with radius 1
# 
# # Red labels indicate sphere with radius 2
# 
# # Green labels indicate sphere with radius 3

# In[77]:


#Preprocessing the data

x_points = []
y_points = []
z_points = []
output = []

x_points = list(x1)+list(x2)+list(x3)
y_points = list(y1)+list(y2)+list(y3)
z_points = list(z1)+list(z2)+list(z3)
output = o1 + o2 + o3
data_points = []
for i in range( len(x_points) ) :
    data_points.append([x_points[i],y_points[i],z_points[i]] )

data_points = StandardScaler().fit_transform(data_points)


# # Liner PCA

# In[114]:


#Linear PCA

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data_points)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])


p1 = list(principalDf['principal component 1'])
p2 = list(principalDf['principal component 2'])

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('Linear PCA', fontsize = 20)
ax.scatter(p1,p2,c=output)


# # FDA

# In[142]:


lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(data_points, output)
lda.explained_variance_ratio_


# In[143]:


fig = plt.figure(figsize = (8,8))
plt.xlabel('LD1',fontsize = 15)
plt.ylabel('LD2',fontsize = 15)
plt.title('LDA',fontsize = 20)
plt.scatter(
    X_lda[:,0],
    X_lda[:,1],
    c=output,
)


# # PCA with polynomial kernel of degree 5

# In[120]:


Kernal = KernelPCA(n_components=2, kernel='poly',degree=5)
k_data = Kernal.fit_transform(data_points)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('Polynomial Kernal PCA with degree 5', fontsize = 20)
ax.scatter(k_data[:,0],k_data[:,1],c=output)


# # PCA with Gaussian kernel

# In[139]:


G_Kernal = KernelPCA(n_components=2, kernel='rbf',gamma=1)
g_data = G_Kernal.fit_transform(data_points)
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('PCA with gaussian kernal', fontsize = 20)
ax.scatter(g_data[:,0],g_data[:,1],c=output)


# In[ ]:




