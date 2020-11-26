#### Implement Naives Bayes Classification
## Test Data: iris.csv or breast-cancer-wisconsin.data

import numpy as np
import pandas as pd
import scipy.stats as stat

## 1. Separate by Class

def sep_by_class(df,class_type,col='class'):
	'''
	Isolates data for each class
	'''
	return df.loc[df[col]==class_type]

## 2. Summarize Dataset

def summarize_data(df,col='class',method=None,*args):
	'''
	Gives mean and standard deviation for all columns. Info argument determines if more information is desired (uses describe method)
	'''
	dg = df.drop(col,axis=1)
	if method == 'describe':
		return dg.describe()#Can use ['mean':'std'], but not desirable for large datasets
	else:
		return pd.concat([dg.mean(),dg.std(),dg.count()],keys=['mean','std','count'],axis=1).T

## 3. Summarize Dataset by Class

def summarize_by_class(df,col='class'):
	frames = []
	classes = df[col].unique()
	for class_type in classes:
		dg = sep_by_class(df,class_type,col)
		frames.append(summarize_data(dg))
	return pd.concat(frames,keys=classes)

## 4. Distributions (Normal primarily used, works for multinomial and bernoulli)

def pdf_distr(x,mu=0,sigma=1,num_trial=2,prob=[.5,.5],distribution='normal',*args):
	'''
	Probability Density functions from scipy.stats used for Bayesian Classifiers. Distributions can be 'normal', 'multinomial', or 'bernoulli'
	'''
	if distribution == 'multinomial' and len(x) > 1:
		return stat.multinomial.pmf(x,num_trial,prob)
	elif distribution == 'bernoulli' and len(x) > 1:
		return stat.bernoulli.pmf(x,prob,mu)
	else: # assumes normal otherwise
		return stat.norm.pdf(x,mu,sigma) 

## 5. Class Probabilities

def calculate_class_prob(summary,x,distr='normal',*args):
	'''
	Summary is output of summarize by class. X is data point (pd.Series). Currently only does normal distribution, which will be default. Will be modified for Multinomial or Bernoulli 
	'''
	total = sum([max(summary.loc[(s,'count')]) for s in summary.index.levels[0]])
	prob = dict()
	for class_values in summary.index.levels[0]:
		prob[class_values] = np.log((max(summary.loc[(class_values,'count')]))/total)
		for s in summary.loc[class_values].columns:
			mu0,sigma0,cnt0 = summary.loc[class_values][s]
			if distr != 'normal': # Incomplete for other distributions, will run normal	
				prob[class_values] += np.log(pdf_distr(x.loc[s],mu=mu0,sigma=sigma0))
			else: 
				prob[class_values] += np.log(pdf_distr(x.loc[s],mu=mu0,sigma=sigma0))
	return prob

## 6. Naive Bayesian Model i.e. P(class = x| A,...,B) = P(A|class=x)*...*P(B|class=x)*P(class=x)
## WARNING: No rescaling, output will NOT be probability

def predict(model,x):
	'''
	Model is output for summarize by class. X is a data point (pd.Series)
	'''
	probabilities = calculate_class_prob(model,x)
	best_label, best_prob = None,-1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return best_label


def naive_bayes(train,test):
	'''
	Train is pd.DataFrame and Test is pd.DataFrame of test points
	'''
	model = summarize_by_class(train)
	predictions = list()
	for r in test.index:
		output = predict(model,test.loc[r])
		predictions.append(output)
	return predictions

## Fit model, training_set is training set with col being class to predict
#model = summarize_by_class(training_set,col='class')



## Predict the label for a given data point
#label = predict(model,x)



## Predict several labels, test is list of data points x
#predictions = naive_bayes(model,test) 
