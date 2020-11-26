### Implement Gradient Descent Linear Regression
## Test Data: tbd

import numpy as np
import pandas as pd


def gradient_descent_regression(dx, dy, m_current=0, b_current=0, epochs=1000, learning_rate=0.0001,*args):
	'''
	m_current,b_current = starting points for m and b, resp. epochs = number of iterations. learning_rate = 
	'''
	N = float(len(dy))
#	if m_current == 0 and dx.shape[1] != 1:
#		m_current = np.array([0]*(dx.shape[1])).reshape(dx.shape[1],1)
#		b_current = np.array([0]*(len(dy))).reshape(dy.shape)
	for _ in range(epochs):
		print(_)
		print(m_current,b_current)
		if m_current is 0:
			y_current = 0*dy
		else:
			y_current = (dx @ m_current ) + b_current
		cost = sum([data**2 for data in (dy-y_current).values]) / N
		m_gradient = -(2/N) * sum((dx * (dy - y_current)).values)
		b_gradient = -(2/N) * sum((dy - y_current).values)
		m_current = m_current - (learning_rate * m_gradient)
		b_current = b_current - (learning_rate * b_gradient)
	return m_current, b_current, cost
