### Implement Gradient Descent Linear Regression
## Test Data: tbd

import numpy as np
import pandas as pd


def gradient_descent_regression(dx, dy, theta=None, epochs=1000, learning_rate=0.0001,*args):
	'''
	theta = starting, random vector, epochs = number of iterations. learning_rate = rate of descent 
	'''
	if theta == None:
		theta = np.random.randn(dx.shape[0],1)
	N = float(len(dy))
	for _ in range(epochs):
		predict = dx @ theta
		theta = theta - (1/N)*learning_rate*(dx.T @ (predidx-dy))
		predict = dx @ theta
	cost = (1/2*N)*np.sum(np.square(predict-dy)) 
	return theta, cost



def stochastic_gradient_descent_regression(dx, dy, theta=None, epochs=1000, learning_rate=0.0001,*args):
	'''
	Stochastic version of gradient descent (incomplete)
	'''
	if theta == None:
		theta = np.random.randn(dx.shape[0],1)
	N = float(len(dy))
	for _ in range(epochs):
		cost = 0.0
		for i in range(N):
			rndm_idx = np.random.permutation(N)
			dx_idx = dx.loc[rndm_idx]
			dy_idx = dy.loc[rndm_idx]
			predict = dx_idx @ theta
			theta = theta - (1/N)*learning_rate*(dx_idx.T @ (predict-dy_idx))
			if _ == (epochs-1):
				cost += (1/2*N)*np.sum(np.square(predict-dy_idx))
	return theta, cost



def batch_gradient_descent_regression(dx, dy, theta=None, batch_size=20, epochs=1000, learning_rate=0.0001,*args):
    '''
    Batch gradient descent (incomplete)
    '''
    if theta == None:
        theta = np.random.randn(dx.shape[0],1)
    N = float(len(dy))
    n_batches = int(N/batch_size)
    for _ in range(epochs):
        cost = 0.0
        for i in range(0,N,batch_size):
            rndm_idx = np.random.permutation(N)
            dx_idx = dx.loc[i:i+batch_size]
            dy_idx = dy.loc[i:i+batch_size]
            predict = dx_idx @ theta
            theta = theta - (1/N)*learning_rate*(dx_idx.T @ (predict-dy_idx))
            if _ == (epochs-1):
                cost += (1/2*N)*np.sum(np.square(predict-dy_idx))
    return theta, cost

	
