#Data curation functions

import os
from csv import reader
import pandas as pd

# Load file
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

# Load data using pandas
def loadfile_pd(filename,colspecs=None,names=None,index_col=None,*args):
	ext = os.path.splitext(filename)[1]
	if ext == '.csv':
		return pd.read_csv(filename,names)
	elif ext == '.xlsx':
		return pd.read_excel(filename,header=0,index_col=0)
	elif ext == '.json':
		return pd.read_json(filename)
	elif ext == '.txt':
		return pd.read_fwf(filename,colspecs,names,index_col)
	else:
		return pd.read_fwf(filename)		
	
# Convert data types in cols

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



