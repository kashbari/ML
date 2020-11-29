### Implement Ordinary Least Squares Regression
## Test data: TBD

import numpy as np
import pandas as pd
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
