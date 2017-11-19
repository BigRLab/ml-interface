from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import os
import pandas as pd
# import pickle
import cPickle as pickle
import numpy as np
import time

TRAIN_PERCENTAGE=0.7

algo_dict={'rf_cla':RandomForestClassifier,
	 'svm_cla':SVC,
	 'dt_cla':DecisionTreeClassifier,
	 'rf_reg':RandomForestRegressor,
	 'svm_reg':SVR,
	 'dt_reg':DecisionTreeRegressor}

def process(param_dict, root_path, job_name):
	# print 'path:', job_path
	# print 'Entered process'
	# data_path=job_path+'\\data\\data'
	data_path=root_path+'\\tmp\\data_'+job_name
	# model_path=job_path+'\\model\\'+job_name+'.pkl'
	model_path=root_path+'\\tmp\\'+job_name+'.pkl'
	# print 'data path:', data_path
	# print param_dict.keys()
	if len(param_dict['header_row'])==0:
		header_row=None
	else:
		header_row=int(param_dict['header_row'])
	# print 'data path:', data_path
	df=pd.read_table(data_path, sep=None, header=header_row, engine='python')
	# print 'data loaded'
	df = df.sample(frac=1).reset_index(drop=True)
	NO_TRAIN=int(TRAIN_PERCENTAGE*df.shape[0])
	NO_TEST=int(df.shape[0]-NO_TRAIN)
	train=df.head(NO_TRAIN)
	test=df.tail(NO_TEST)
	algo_name=param_dict['algorithm']+'_'+param_dict['train_type']
	#algorithm, train_type, job_name, y_col, file, train_type
	y=df[int(param_dict['y_col'])]
	y_map={}
	for ix, y_val in enumerate(y.unique()):
		y_map[y_val]=ix
	y=y.apply(lambda row: y_map[row])
	y_train=df[int(param_dict['y_col'])].head(NO_TRAIN)
	y_test=df[int(param_dict['y_col'])].tail(NO_TEST)
	X=df.drop(int(param_dict['y_col']), axis=1)
	X_train=X.head(NO_TRAIN)
	X_test=X.tail(NO_TEST)
	model=algo_dict[algo_name]()
	model.fit(X_train, y_train)
	predictions=model.predict(X_test)
	metric_value=-1
	if param_dict['train_type']=='cla':
		acc=accuracy_score(y_test, predictions)
		metric_value=acc
		# print 'acc', metric_value
	else:
		rmse=np.sqrt(mean_squared_error(y_test, predictions))
		metric_value=rmse
		# print 'rmse', metric_value
	with open(model_path, 'w') as f:
		pickle.dump(model, f)
		# print 'model saved'
	return root_path, metric_value

def make_predictions(model_path, data_path, header):
	X=pd.read_table(data_path, sep=None, header=header, engine='python')
	#with open(model_path, 'rb') as f:
	model=pickle.load(open(model_path, 'r'))
	pred=model.predict(X)
	return pred