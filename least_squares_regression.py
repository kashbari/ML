### Implement Ordinary Least Squares Regression
## Test data: TBD

import numpy as np
import pandas as pd
from mxnet import autograd, np, npx
import random
import scipy.stats as stat
npx.set_np()


def normal_pdf(x,mu=0,sigma=1,*args):
    """ Returns Gaussian pdf evaluated at x with loc-scal params mu and sigma """
    return stat.norm.pdf(x,mu,sigma)

### Implementation using MXNET




## Implementation using PyTorch









## Implementation using statsmodels of
from statsmodels import tools
import statsmodels.formula.api as sm


def ols_data(dx,dy):
	'''
	dy and dx are pd.DataFrames for dep and indep variable data. Combines into one DataFrame for regression
	'''
	return pd.concat([dx,dy],axis=1)

def least_squares_regression(df,col_dep='last',*args):
	if col_dep == 'first':
		form_str = f"{df.columns[0]} ~ "+" + ".join(df.columns[1:]) 
	else:
		form_str = f"{df.columns[-1]} ~ "+" + ".join(df.columns[:-1])
	return sm.ols(formula = form_str,data=df).fit()
	
## result = ols.regression(df)
## result.params returns parameters (intercept, etc.)
## result.summary() for more regression results







