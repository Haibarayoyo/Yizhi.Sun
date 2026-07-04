#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import package
import numpy as np


# In[2]:


# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of activation function
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# In[3]:


# Loss function
def mse_loss(y_true, y_predicted):
    return np.mean(np.square(y_true - y_predicted))

# Derivative of loss function
def mse_loss_derivative(y_true, y_predicted):
    return 2 * (y_predicted - y_true) / y_true.size


# In[4]:


# Input data
X = np.array([[0.5, 0.2]])
# True outcome
y_true = np.array([[0.7]])

# Initial parameters of the hidden layer
W_h = np.array([[0.05, 0.1],
                [0.02, 0.08]])
b_h = np.array([[0.01, 0.02]])

# Initial parameters of the output layer
W_o = np.array([[0.1],
                [0.03]])
b_o = np.array([[0.05]])


# ### Step 1: forward pass

# In[5]:


# np.dot() here performs matrix multiplication since X and W_h are both 2-D arrays.
Z_h = np.dot(X, W_h) + b_h
A_h = sigmoid(Z_h)

Z_o = np.dot(A_h, W_o) + b_o
A_o = sigmoid(Z_o)


# ### Step 2: calculate errors

# In[11]:


loss = mse_loss(y_true, A_o)
print("Loss:", loss)


# ### Step 3: backward pass

# In[12]:


dLoss_dA_o = mse_loss_derivative(y_true, A_o)
dA_o_dZ_o = sigmoid_derivative(Z_o)
delta_o = dLoss_dA_o * dA_o_dZ_o

grad_W_o = np.dot(A_h.T, delta_o)
grad_b_o = np.sum(delta_o, axis=0, keepdims=True)

dLoss_dA_h = np.dot(delta_o, W_o.T)
dA_h_dZ_h = sigmoid_derivative(Z_h)
delta_h = dLoss_dA_h * dA_h_dZ_h

grad_W_h = np.dot(X.T, delta_h)
grad_b_h = np.sum(delta_h, axis=0, keepdims=True)


# ### Step 4: update parameters

# In[13]:


learning_rate = 0.1

W_o -= learning_rate * grad_W_o
b_o -= learning_rate * grad_b_o
W_h -= learning_rate * grad_W_h
b_h -= learning_rate * grad_b_h

# Print updated parameters
print("Updated W_o:", W_o)
print("Updated b_o:", b_o)
print("Updated W_h:", W_h)
print("Updated b_h:", b_h)

