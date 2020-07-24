#!/usr/bin/env python
# coding: utf-8

# In[113]:


import numpy as np
from hmmlearn.hmm import MultinomialHMM


# In[114]:


#cleaning the data
# Task-1
fd1 = open("hmm-train.txt","r");
s = fd1.read();
s = s.replace('\n', '').replace('\r', '')
punctuations = '''!1234567890()-—[]{};:'"\,<>./?@#$%^&*_~''';
for x in punctuations:
    s = s.replace(x,'')
s = s.upper()
print(s)
#data has been preprocessed as a stream of characters and it has been cleaned by removing puncutations and spaces


# In[115]:


# Task-2
# building a simple natural hmm by giving same priority to two states
# the state1 consitutes "ABCDEFGHIJKLM"
# the other state2 contains the remaining characters
state1 = "ABCDEFGHIJKLM"
transition_prob = np.zeros((2,2))
emission_prob = np.zeros((27,2))
transition_sum = np.zeros((2,2))
emission_sum = np.zeros((27,2))
initial_state = 1
emission_sum[ord('T')-ord('A')][initial_state]=1
present_state = initial_state
for i in s:
    map1 = ord(i) - ord('A')
    prev = present_state
    if i == ' ' :
        map1 = 26
    if i in state1:
        present_state = 0
        transition_sum[prev][present_state]=transition_sum[prev][present_state]+1
    else :
        present_state = 1
        transition_sum[prev][present_state]=transition_sum[prev][present_state]+1
    emission_sum[map1][present_state] = emission_sum[map1][present_state]+1
    
transition_prob[0][0] = transition_sum[0][0] / (transition_sum[0][1]+transition_sum[0][0])
transition_prob[0][1] = 1 - transition_prob[0][0]
transition_prob[1][0] = transition_sum[1][0] / (transition_sum[1][1]+transition_sum[1][0])
transition_prob[1][1] = 1 - transition_prob[1][0]

emission_prob[:,0] = np.divide(emission_sum[:,0],sum(emission_sum[:,0]))
emission_prob[:,1] = np.divide(emission_sum[:,1],sum(emission_sum[:,1]))
print("A,B,C,D..... are hashed as 0,1,2,3,4....")
print("Tranisiton Probability of my proposed model")
print("transition_prob[i][j] indicate the probability of going ith state to jth state")
print(transition_prob)
print("Emission Probability of my proposed model")
print("emission_prob_prob[i][j] indicate the probability of giving ith charcter given in state j")
print(emission_prob)

# Printing the seven most likely characters
print("For state 0, the seven most likely characters are\n")
arg1 = emission_prob[:,0].argsort()[-7:][::-1]
for i in arg1 : 
    if(i == 26):
        print("Space")
    else:
        print(chr(i+ord('A')))
print("For state 1, the seven most likely characters are\n")
arg1 = emission_prob[:,1].argsort()[-7:][::-1]
for i in arg1 : 
    if(i == 26):
        print("Space")
    else:
        print(chr(i+ord('A')))


# In[91]:


### preparing the testdata as the list of numbers
## task -3

dummy_data = []
for i in s:
    if i == ' ' :
        dummy_data.append(26)
    else :
        dummy_data.append( ord(i)-ord('A') )
        
training_data = np.array(dummy_data)
training_data = training_data.reshape((training_data.shape[0],1))

### hmm model
hmm_model = MultinomialHMM(n_components=2,n_iter=500, tol=0.01, verbose=False)
hmm_model.fit(training_data)
print(hmm_model.monitor_)

print("Tranisition probalitity of this model is \n")
print(hmm_model.transmat_)
print("\n")
print("Emission probalitity of this model is \n")
print(np.transpose(hmm_model.emissionprob_))
print("\n")
## the seven most probable characters
transition_prob1 = transition_prob
emission_prob1 = np.transpose(hmm_model.emissionprob_)

print("For this trained model, the seven most likely charcters are\n")

print("For state 0, the seven most likely characters are\n")
arg1 = emission_prob1[:,0].argsort()[-7:][::-1]
for i in arg1 : 
    if(i == 26):
        print("Space")
    else:
        print(chr(i+ord('A')))
print("For state 1, the seven most likely characters are\n")
arg1 = emission_prob1[:,1].argsort()[-7:][::-1]
for i in arg1 : 
    if(i == 26):
        print("Space")
    else:
        print(chr(i+ord('A')))
print("By observing the most likely charcters it seems like I should have used vowels and consonants as two sepaerate states")


# In[109]:


#task 4
evaluate_hmm_model = hmm_model.score(training_data)

print("The score of the inbuilt trained model is")
print(evaluate_hmm_model)
print("\n")

hmm_natural_model = MultinomialHMM(n_components=2)
hmm_natural_model.transmat_ = transition_prob
hmm_natural_model.emissionprob_ = np.transpose(emission_prob)
hmm_natural_model.startprob_ = np.array([0,1])
evaluate_hmm_natural = hmm_natural_model.score(training_data)
print("The score of my designed natural hmm is")
print(evaluate_hmm_natural)
print(hmm_natural_model.monitor_)
print("\n")
print("Since, the score of the inbulit hmm model is more than the natural model")
print("Therefore the performance of inbuilt hmm model is good\n")

print("Training the natural hmm")
hmm_natural_model1 = MultinomialHMM(n_components=2,n_iter=500)
hmm_natural_model1.transmat_ = transition_prob
hmm_natural_model1.emissionprob_ = np.transpose(emission_prob)
hmm_natural_model1.startprob_ = np.array([0,1])
hmm_natural_model1.fit(training_data)
print(hmm_natural_model1.monitor_)
print("After training, the score of natural hmm is,", hmm_natural_model1.score(training_data))


# In[110]:


#task 5
fd1 = open("hmm-test.txt","r");
s = fd1.read();
s = s.replace('\n', '').replace('\r', '')
punctuations = '''Â!1234567890()-—[]{};:'"\,<>./?@#$%^&*_~€”''';
for x in punctuations:
    s = s.replace(x,'')
s = s.upper()

dummy_data = []
for i in s:
    if ord(i) - ord('A') > 26 or ord(i)-ord('A') < 0 :
        continue
    if i == ' ' :
        dummy_data.append(26)
    else :
        dummy_data.append( ord(i)-ord('A') )
test_data = np.array(dummy_data)
test_data = test_data.reshape((test_data.shape[0],1))

print("The score of the test_data on natural_hmm is ", hmm_natural_model.score(test_data))
print("The score of the test_data on inbuilt hmm is", hmm_model.score(test_data))
print("The score of the test_data on natural_hmm is ", hmm_natural_model1.score(test_data))

print("Its quite suprising that both models are upto the same level and it suggests that on trained data,")
print("the inbuilt model is sligthy been overfitted\n")


# In[ ]:




