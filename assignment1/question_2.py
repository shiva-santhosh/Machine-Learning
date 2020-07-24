#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd 
import numpy as np 
import csv
from random import shuffle
import matplotlib.pyplot as plt 


# In[6]:


csv_data  = pd.read_csv("train.csv")
train_data = np.array(csv_data)
W = np.zeros((11))
W1 = np.zeros((11))

lb = 0.7
x = []
y = []
for i in train_data :
    x.append(i[0])
    y.append(i[1])
    
plt.scatter(x,y)

np.random.shuffle(train_data)

def get_vector(x) :
    x1 = np.zeros((11))  
    for i in range(11) :
        x1[i] = x**i
    return x1 

def ridge_regression(X,Y,X1,Y1,lb) :
    A = np.linalg.pinv( np.dot(X.transpose(),X) + lb*np.identity(11)) 
    B = np.dot(X.transpose(),Y)
    W = np.dot(A,B)
    train_loss = np.sum((np.dot(X,W)-Y)**2) /len(X)
    validation_loss = np.sum((np.dot(X1,W)-Y1)**2) /len(X1)
    return W,train_loss,validation_loss

def lasso_regression(X, Y, X1, Y1, lb) :
    A1 = np.linalg.pinv( np.dot(X.transpose(),X) )
    B1 = np.dot(X.transpose(),Y)
    C1 = lb/2 * (np.sign( np.dot(A1,B1)))
    W1 = np.dot(A1, (B1-C1))
    train_loss = np.sum((np.dot(X,W1)-Y)**2) /len(X)
    validation_loss = np.sum((np.dot(X1,W1)-Y1)**2) /len(X1)
    return W1,train_loss,validation_loss


# In[7]:


X = np.zeros((len(train_data),11))
Y = np.zeros((len(train_data))) 

for i in range(len(train_data)) :
    X[i] = get_vector(train_data[i][0]) 
    Y[i] = train_data[i][1]
    

K = 3 

X1 = X[:17]
X2 = X[17:35]
X3 = X[35:]

Y1 = Y[:17]
Y2 = Y[17:35]
Y3 = Y[35:]

model_1_1 = ridge_regression(np.concatenate((X1,X2)), np.concatenate((Y1,Y2)), X3, Y3, lb  )
model_1_2 = ridge_regression(np.concatenate((X1,X3)), np.concatenate((Y1,Y3)), X2, Y2, lb  )
model_1_3 = ridge_regression(np.concatenate((X2,X3)), np.concatenate((Y2,Y3)), X1, Y1, lb  )

avg_training_loss,avg_validation_loss = (model_1_1[1]+model_1_2[1]+model_1_3[1])/3 , (model_1_1[2]+model_1_2[2]+model_1_3[2])/3

print("For ridge regression model")
print(avg_training_loss,avg_validation_loss)


model_2_1 = lasso_regression(np.concatenate((X1,X2)), np.concatenate((Y1,Y2)), X3, Y3, lb  )
model_2_2 = lasso_regression(np.concatenate((X1,X3)), np.concatenate((Y1,Y3)), X2, Y2, lb  )
model_2_3 = lasso_regression(np.concatenate((X2,X3)), np.concatenate((Y2,Y3)), X1, Y1, lb  )

avg_training_loss1,avg_validation_loss1 = (model_2_1[1]+model_2_2[1]+model_2_3[1])/3 , (model_2_1[2]+model_2_2[2]+model_2_3[2])/3
print("For lasso regression model")
print(avg_training_loss1,avg_validation_loss1)


# In[8]:


# out of two models Ridge regression is giving less training_loss and validation_loss therby performing better
# therefore parameters obtained using Ridge regression are used


model_selected = ridge_regression(X,Y,X,Y,lb)

W = model_selected[0]
 
#The submission.csv file contains 2 columns first column is input train data and 2 column is predicted model

csv_data  = pd.read_csv("testX.csv")
test_data = np.array(csv_data)
with open('submission.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for i in range(len(test_data)) :
        val1 = test_data[i][0]
        val2 = np.dot (W.transpose(), get_vector(test_data[i][0]) )
        writer.writerow( (str(val1),str(val2)) )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




