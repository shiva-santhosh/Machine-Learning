#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 


# In[2]:


fd1 = open("iris.data").readlines()
z1  = [x.rstrip() for x in fd1]
iris_data = []
for i in range(len(z1)) :
    iris_data.append([x.strip() for x in z1[i].split(',')])
iris_data.pop()


# In[3]:


map1 = {}
map1["Iris-setosa"] = "1"
map1["Iris-versicolor"] = "2"
map1["Iris-virginica"] = "3"


# In[4]:


pretty_output = []
fd2 = open("iris-svm-input.txt","w") 
for i in range(0,len(iris_data)) :
    str1 = ""
    str1 = str1 + map1[ iris_data[i][4] ] + " "
    for j in range(4) :
        if(float(iris_data[i][j]) != 0) :
            str1 = str1 + str(j+1) + ":" + iris_data[i][j] + " "
    str1 = str1 + "\n "
    pretty_output.append(str1) 
    
fd2.writelines(pretty_output)


# In[ ]:




