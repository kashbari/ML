### Implement k-fold cross-validation 
## Choices for k: 
## 1. Representative: choose such that each train/test group is large enough to be statistically representative
## 2. k = 10 or 5: found through experimentation to generally result in low bias and modest variance
## 3. k = n: n is size of dataset, gives each test sample opportunity to be used in hold out dataset;leave-one-out cross-validation

## RMK: k larger => bias becomes smaller

import numpy as np
import pandas as pd
from random import *
import copy

## 1. Evaluation Metric

def eval_metric(actual,predicted,metric='accuracy',class_positive=None,*args):
	'''
	Evaluates how good results are. Metric default is accuracy. Other metrics to use: precision, recall, and f1 (F1-score or harmonic mean). Class_positive is for class to determine how many true positives. 
	'''
	if class_positive == None:
		a = actual[0]
	else:
		a = class_positive
	if metric == 'precision':
		total_pos = sum([a == predicted[i] for i in range(len(predicted))])
		true_pos = sum([(a == actual[i]) and (actual[i] == predicted[i]) for i in range(len(predicted))])
		return (true_pos*100.0)/total_pos 
	elif metric == 'recall':
		true_pos = sum([(a == actual[i]) and (actual[i] == predicted[i]) for i in range(len(predicted))])
		false_neg = sum([(a == actual[i]) and (actual[i] != predicted[i]) for i in range(len(predicted))])
		return (true_pos*100.0)/(true_pos+false_neg)
	elif metric == 'f1':
		r = eval_metric(actual,predicted,metric='recall',class_positive=a)
		p = eval_metric(actual,predicted,metric='precision',class_positive=a)
		return (2*p*r)/(p+r)
	else: # default assumes accuracy
		correct = sum([predicted[i] == actual[i] for i in range(len(actual))])
		return (correct*100.0)/len(actual)

## 2. Cross-Validation Split

#n_folds = 5
def cross_valid_split(df,n_folds=5,*args):
	df_split = list()
	fold_size = int(len(df)/n_folds)
	for _ in range(n_folds):
		idx = sample(df.index,fold_size)
		df_split.append(df.loc[idx])
	return df_split

## 3. Evaluaton of Algorithm

def cross_valid_eval_alg(df,algorithm,n_folds=5,*args):
	folds = cross_valid_split(df,n_folds)
	scores = list()
	for f in folds:
		train_set = copy.deepcopy(folds)
		f1 = [j for j in range(len(train_set)) if f.equals(train_set[j])]
		del train_set[f1[0]]
		train_set = pd.concat(train_set)
		train_set.index = list(range(len(train_set)))
		test_set = copy.deepcopy(f)
		test_set.index = list(range(len(test_set)))
		predicted = algorithm(train_set,test_set)
		actual = [test_set.loc[(r,'class')] for r in test_set.index]
		accuracy = eval_metric(actual,predicted)
		scores.append(accuracy)
	return scores
