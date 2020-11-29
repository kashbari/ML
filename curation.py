#### Data curation functions

import os
from csv import reader
import numpy as np
import pandas as pd

## Load file (Non-pandas) DEPRECATED
def loadfile(filename):
	extension = os.path.splitext(filename)[1]
	if extension == '.csv':
		dataset = list()
		with open(filename, 'r') as file:
			csv_read = read(file)
			for row in csv_reader:
				if not row:
					continue
				dataset.append(row)
		return dataset
	elif extension == '.txt':
		dataset = eval(open(filename).read())
		return dataset

## Pandas load
## dataset = pd.read_csv("filename.csv",sep=",")
## sep = "," for comma, "\t" for tab, ";" for semi-colon,


## Load data using pandas
def loadfile_pd(filename,col_names=None,colspecs=None,idx_col=None,*args):
	'''
	Load file into pandas dataframe. Assumes .csv ext
	'''
	ext = os.path.splitext(filename)[1]
	if ext == '.data':
		return pd.read_csv(filename,names=col_names,index_col=idx_col)
	elif ext == '.xlsx':
		return pd.read_excel(filename,header=0,index_col=0)
	elif ext == '.json':
		return pd.read_json(filename)
	elif ext == '.txt':
		return pd.read_fwf(filename,colspecs,names=col_names,index_col=idx_col)
	else:
		return pd.read_csv(filename,names=col_names)		
	
## Convert data types in cols (non-pandas) DEPRECATED/INCOMPLETE
def convert_data(data,idx,data_type2,axis='col'):
	if axis == col:
		if data_type2 == 'float':
			for row in dataset:
				row[idx] = float(row[idx].strip())
		if data_type2 == 'int':
			for row in dataset:
				row[idx] = int(row[idx].strip())
	else:
		print('TBD')
	return	

## df.dtypes to see data types
## df = df.astype('int64') to convert to type int64; or dictionary for particular cols
## df = df.drop( [list_of_row_indicies] )
## df = df[~df.index.duplicated(keep='first')] to remove duplicates by keeping first

## Initial Steps for Data Cleaning:

def clean_data_prelim(df,remove_duplicates=False,keep='last',change_class_type=False,numeric_data=None,drop_rows=None,verbose=True,*args):
	'''
	Cleans data preliminarily so all data (except possibly class) is of appropriate type and removing duplicate entries. verbose (Boolean) returns summary. remove_duplicates and keep determine which duplicates to remove. change_class_type (Boolean) and numeric_data (list of cols for to_numeric) determine new data types. drop_rows (list) will drop rows.
	''' 
	if clean_data_prelim.func_defaults != (remove_duplicates,keep,change_class_type,numeric_data,drop_rows,verbose):
		if remove_duplicates == True and keep == 'last':
			print('############## Removing Duplicates (Keep last).... #########')
			df = df[~df.index.duplicated(keep='last')]
#			print('done.')
		elif remove_duplicates == True and keep == 'first':
			print('############## Removing Duplicates (Keep first).... #########')
			df = df[~df.index.duplicated(keep='first')]
#			print('done.')	
		if change_class_type == True:
			print('########## Changing Data Types.... ###########')
			df[numeric_data] = pd.to_numeric(df[numeric_data],errors='coerce')
#			try:
#				df = df.astype(dict_data_types)
#			except ValueError:
#				df = pd.to_numeric(df[dict_data_types.keys()],errors='coerce')
		if verbose == True:
			clean_data_prelim(df)
		return df
	else:
		if verbose == True:
			print('############# DataFrame: ##############')
			print(df.head())
			print('############# Data Types in DataFrame: ###############')
			print(df.dtypes)
			print('############# Number of Data points: ###############')
			print(len(df))
			print('############# Number of repeat elements: ###############')
			print(len(df) - len(df.index.unique()))
			class_O = list()
			class_str = list()
			class_unknown = list()
			for idx in df.dtypes.index:
				if df.dtypes[idx] == 'O':
					class_O.append(idx)
				elif df.dtypes[idx] == 'str':
					class_str.append(idx)
				elif df.dtypes[idx] != 'int':
					class_unknown.append(idx)
			if len(class_O) != 0:
				print('###### Object Datatype: #######')
				print(class_O)
			if len(class_str) != 0:
				print('###### String Datatype: #######')
				print(class_str)
			if len(class_unknown) != 0:
				print('###### Other Datatype: ######')
				print(class_unknown)
	return		
