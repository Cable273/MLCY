#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from progressbar import ProgressBar
    
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

    new_data[n,4:] = c2
    # new_data[n,4:] = c3
	
# data = new_data

np.random.seed(1)

# rms_scores=[]
# r2_scores=[]
# acc_scores=[]
hodge_cut_index=np.arange(2,4)
print(hodge_cut_index)
for trial in np.arange(1,11):
	metrics_data = np.zeros(4)
	np.random.shuffle(data)
	for index in range(0,np.size(hodge_cut_index,axis=0)):
		hodge_cut=hodge_cut_index[index]
		print(hodge_cut)
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
		# n = np.max(data[:,2])
		y_train,y_test = train_data[:,2],test_data[:,2]

		# train_data,test_data = train_test_split(0.2,data)
		# x_train,x_test = train_data[:,4:],test_data[:,4:]
		# y_train,y_test = train_data[:,2],test_data[:,2]

		from svm_functions import svm_dual,gen_model,b_from_model,bulk_svm_eval,svm_eval
		from svm_functions import svm_dual_reg,gen_model_reg,bulk_svm_reg_eval
		from svm_kernels import lin_kernel,poly_kernel,gauss_kernel

		kernel,std = 'gauss_kernel',2.74
		alphas = svm_dual_reg(x_train,y_train,kernel,std,10,0.01)
		model=gen_model_reg(alphas,x_train)
		# np.savetxt('model',model)
		# model=np.loadtxt('./model')

		#eval regressor with no shift
		train_predict = bulk_svm_reg_eval(x_train,model,kernel,std,0)
		test_predict = bulk_svm_reg_eval(x_test,model,kernel,std,0)

		#find shifts from average distance
		b_train = np.mean(y_train - train_predict)
		b_test = np.mean(y_test - test_predict)

		#eval regressor with correct shift
		train_predict = bulk_svm_reg_eval(x_train,model,kernel,std,b_train)
		test_predict = bulk_svm_reg_eval(x_test,model,kernel,std,b_test)

		from h11_freq_plot_functions import plot_h11_freq
		# plot_h11_freq(test_predict,y_test,0.2)

		from performance_metrics import acc_metrics, rms_error, reg_acc, r2 ,F_reg, acc_metrics_reg

		y_train = y_train
		train_predict =  train_predict
		y_test = np.round(y_test)
		test_predict = np.round(test_predict)

		# for n in range(0,np.size(test_predict,axis=0)):
				# print(test_predict[n],y_test[n])

		rms_score = rms_error(test_predict,y_test)
		r2_score = r2(test_predict,y_test)
		acc_score = reg_acc(test_predict,y_test,0.5)

		data_to_save = np.array((hodge_cut,rms_score,r2_score,acc_score))
		metrics_data= np.vstack((metrics_data,data_to_save))
		print(data_to_save)

		test_data_to_save = np.array((y_test,test_predict)).transpose()
		print(test_data_to_save)
		np.savetxt("h11,cicy_squared,svm,hodge_cut_predictions,"+str(hodge_cut)+","+str(trial),test_data_to_save)
		# np.savetxt("h11,cicy_cubed,svm,hodge_cut_predictions,"+str(hodge_cut)+","+str(trial),test_data_to_save)

	metrics_data = np.delete(metrics_data,0,axis=0)
	print(metrics_data)
	np.savetxt("h11,cicy_squared,svm,metrics,trial,"+str(trial),metrics_data)
	# np.savetxt("h11,cicy_cubed,svm,metrics,trial"+str(trial),metrics_data)

		# def hist(x,label_str):
				# x0=np.sort(np.unique(x))
				# freq=np.zeros(np.size(x0))
				# for n in range(0,np.size(x0,axis=0)):
						# for m in range(0,np.size(x,axis=0)):
								# if x[m] == x0[n]:
										# freq[n] = freq[n] + 1
				# plt.plot(x0,freq,label=str(label_str))
				# plt.xlabel("h11")
				# plt.ylabel("Class size")
				# # plt.show()
		# hist(test_predict,"x<="+str(hodge_cut)+" prediction")
		# np.save("svm,test_acc,hodge_cut,"+str(hodge_cut),test_predict)
		# h11=data[:,2]
		# hist(h11,"h11 True distribution")
# plt.legend()
# plt.show()


# rms_scores = np.append(rms_scores,rms_error(test_predict,y_test))
# r2_scores = np.append(rms_scores,r2(test_predict,y_test))
# acc_scores = np.append(rms_scores,reg_acc(test_predict,y_test,0.5))

# np.save("keras,hodge,h_cut,test_rms",rms_scores)
# np.save("keras,hodge,h_cut,test_r2",r2_scores)
# np.save("keras,hodge,h_cut,test_acc",acc_scores)
# print(rms_scores)

# plt.plot(np.arange(0,np.size(rms_scores)),rms_scores,label="rms")
# plt.plot(np.arange(0,np.size(rms_scores)),r2_scores,label="r2")
# plt.plot(np.arange(0,np.size(rms_scores)),acc_scores,label="acc")
# plt.legend()
# plt.show()

