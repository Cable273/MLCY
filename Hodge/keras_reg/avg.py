#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

def avg(split,trials):
	temp = np.zeros((trials,6))
	for trial in range(0,trials):
		temp[trial,:] = np.loadtxt('./acc_data/split_'+str(split)+'/hodge_acc_split,'+str(split)+',trial,'+str(trial+1))
	mean = np.mean(temp,axis=0)
	std = np.power(np.var(temp,axis=0),0.5)
	return np.array((mean,std)).transpose()

no_splits=int(0.95/0.05)
# no_splits=2
print(avg(0.9,100))

avg_test = np.zeros((no_splits,7))
avg_train = np.zeros((no_splits,7))

for count in range(0,no_splits):
        split  = np.round((count+1)*0.05,decimals=2)
        avg_train[count,0], avg_train[count,1:4], avg_train[count,4:]= split, avg(split,100)[0:3,0],avg(split,100)[0:3,1]
        avg_test[count,0], avg_test[count,1:4], avg_test[count,4:]= split, avg(split,100)[3:,0],avg(split,100)[3:,1]
print(avg_train)
# print(avg_test)

# # np.savetxt('hodge_nn_reg_test_LC_avg',avg_test)
# # np.savetxt('hodge_nn_reg_train_LC_avg',avg_train)
