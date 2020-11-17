#Data curation functions

import os
from csv import reader
import pandas as pd

# Load file
def loadfile(filename):
	type = os.path.splitext(filename)[1]
	if type == 'csv':
		dataset = list()
		with open(filename, 'r') as file:
			csv_read = read(file)
			for row in csv_reader:
				if not row:
					continue
				dataset.append(row)
		return dataset
	elif type == 'txt':
		dataset = eval(open(filename).read())
		return dataset
	
# Convert data type

def convert_data(data,axis='col',idx,data_type2):
	if axis == col:
		if data_type2 == 'float':
			for row in dataset:
				row[idx] = float(row[idx].strip())
		if data_type2 == 'int':
			for row in dataset:
				row[idx] = int(row[idx].strip())
	else:
		print('Not done yet')
	return	