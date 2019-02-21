#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from progressbar import ProgressBar

import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import MaxPooling1D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.regularizers import l1
from keras import optimizers
from keras import regularizers
from keras.layers import GaussianNoise
from keras.layers import BatchNormalization
from keras.initializers import glorot_uniform

#Create network architechture
network = Sequential()
network.add(Dense(939,input_dim=180))
network.add(Activation('relu'))
network.add(Dropout(0.324521))
network.add(Dense(1))
network.add(Activation('sigmoid'))

network.save_weights('network_init.h5')

# val_acc_stop = EarlyStopping(monitor='val_acc',min_delta=0,patience=5,verbose=1,mode='auto')
val_loss_stop = EarlyStopping(monitor='val_loss',min_delta=0,patience=10,verbose=1,mode='auto')
    
##Main##
#Import data
def train_test_split(test_size,data):
#test_size = fraction of total data to be test sample
    length_train = int(np.size(data,axis=0)-np.floor(test_size*np.size(data,axis=0)))
    train_data=data[0:length_train,:]
    test_data=data[length_train:,:]
    return np.array((train_data,test_data))

data=pd.read_csv('../data',sep=",",skiprows=[0],header=None)
data=np.asfarray(data,float)

#append square and cube of CICY matrix to trainind data
# new_data = np.zeros((np.size(data,axis=0),4+12*15*3))
new_data = np.zeros((np.size(data,axis=0),4+12*15))
for n in range(0,np.size(data,axis=0)):
    cicy_vector = data[n,4:]
    c2 = cicy_vector**2
    c3 = np.power(cicy_vector,3)

    new_data[n,0:4] = data[n,0:4]
    new_data[n,4:] = c2
    # new_data[n,4:] = c3
	
data = new_data

np.random.seed(1)


hodge_cut_index=np.arange(2,4)
for trial in np.arange(1,11):
	np.random.shuffle(data)
	metrics_data = np.zeros(4)
	for index in range(0,np.size(hodge_cut_index,axis=0)):
		hodge_cut = hodge_cut_index[index]
		train_data = np.zeros(np.size(data,axis=1))
		test_data = np.zeros(np.size(data,axis=1))
		pbar=ProgressBar()
		for n in pbar(range(0,np.size(data,axis=0))):
				if data[n,2] <= hodge_cut:
						train_data = np.vstack((train_data,data[n]))
				else:
						test_data = np.vstack((test_data,data[n]))
		train_data = np.delete(train_data,0,axis=0)
		test_data = np.delete(test_data,0,axis=0)

		print("Orig split:")
		print(np.size(train_data,axis=0))
		print(np.size(test_data,axis=0))

		#take 10% from test data
		dim = int(np.size(test_data,axis=0)/10)
		train_data = np.vstack((train_data,test_data[:dim,:]))
		test_data = np.delete(test_data,np.arange(0,dim),axis=0)
		np.random.shuffle(train_data)

		print("New split:")
		print(np.size(train_data,axis=0))
		print(np.size(test_data,axis=0))

		x_train,x_test = train_data[:,4:],test_data[:,4:]
		n_max = np.max(data[:,2])
		y_train,y_test = train_data[:,2]/n_max,test_data[:,2]/n_max

		# class_balance = np.vstack((class_balance,np.array((np.size(x_train,axis=0),np.size(x_test,axis=0)))))

		# train_data,test_data = train_test_split(0.2,data)
		# x_train,x_test = train_data[:,4:],test_data[:,4:]
		# n = np.max(data[:,2])
		# y_train,y_test = train_data[:,2]/n,test_data[:,2]/n

		# x_train=np.reshape(x_train,(np.size(x_train,axis=0),12,15,1))
		# x_test=np.reshape(x_test,(np.size(x_test,axis=0),12,15,1))

		from keras import backend as K
		def r2_met(y_true,y_pred):
				SS_res =  K.sum(K.square( y_true-y_pred )) 
				SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
				return ( 1 - SS_res/(SS_tot + K.epsilon()) )

		network.load_weights('network_init.h5')
		network.compile(loss='mean_absolute_error',optimizer='adam',metrics=[r2_met])
		history = network.fit(x_train,y_train,batch_size=32,epochs=1000,verbose=1,validation_data=(x_test,y_test),callbacks=[val_loss_stop])
		# network.save('./network.h5')
		# network.load_weights('./network.h5')

		#rescale by n
		train_predict = n_max*network.predict(x_train,verbose = 1)[:,0]
		test_predict = np.round(n_max*network.predict(x_test,verbose = 1)[:,0])
		y_train = n_max*y_train
		y_test = np.round(n_max*y_test)

		from performance_metrics import acc_metrics,rms_error,reg_acc,r2,F_reg
		for n in range(0,np.size(test_predict,axis=0)):
				print(y_test[n],test_predict[n])

		# for n in range(0,np.size(train_predict,axis=0)):
				# print(y_train[n],train_predict[n])
		rms_score = rms_error(test_predict,y_test)
		r2_score = r2(test_predict,y_test)
		acc_score = reg_acc(test_predict,y_test,0.5)

		data_to_save = np.array((hodge_cut,rms_score,r2_score,acc_score))
		metrics_data= np.vstack((metrics_data,data_to_save))
		print(data_to_save)

		test_data_to_save = np.array((y_test,test_predict)).transpose()
		print(test_data_to_save)
		np.savetxt("h11,cicy_squared,keras,hodge_cut_predictions,"+str(hodge_cut)+","+str(trial),test_data_to_save)
		# np.savetxt("h11,cicy_cubed,keras,hodge_cut_predictions,"+str(hodge_cut)+","+str(trial),test_data_to_save)

	metrics_data = np.delete(metrics_data,0,axis=0)
	print(metrics_data)
	np.savetxt("h11,cicy_squared,keras,metrics,trial"+str(trial),metrics_data)
	# np.savetxt("h11,cicy_cubed,keras,metrics,trial"+str(trial),metrics_data)


# # rms_scores = np.append(rms_scores,rms_error(test_predict,y_test))
# # r2_scores = np.append(r2_scores,r2(test_predict,y_test))
# # acc_scores = np.append(acc_scores,reg_acc(test_predict,y_test,1))
# # plt.plot(np.arange(0,np.size(rms_scores)),rms_scores,label="rms")
# # plt.plot(np.arange(0,np.size(rms_scores)),r2_scores,label="r2")
# # plt.plot(np.arange(0,np.size(rms_scores)),acc_scores,label="acc")
# # plt.legend()
# # plt.show()

# # np.save("keras,hodge,h_cut,test_rms",rms_scores)
# # np.save("keras,hodge,h_cut,test_r2",r2_scores)
# # np.save("keras,hodge,h_cut,test_acc",acc_scores)
