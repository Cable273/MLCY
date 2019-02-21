#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.special import erfinv

svm_train_data = np.loadtxt('./hodge_svm_train_LC_avg')
svm_test_data = np.loadtxt('./hodge_svm_test_LC_avg')

nn_reg_train_data = np.loadtxt('./hodge_nn_reg_train_LC_avg')
nn_reg_test_data = np.loadtxt('./hodge_nn_reg_test_LC_avg')

nn_class_train_data = np.loadtxt('./hodge_nn_class_train_LC_avg')
nn_class_test_data = np.loadtxt('./hodge_nn_class_test_LC_avg')

def max_reg(data):
    max_acc_index = np.argmax(data[:,1])
    max_rms_index = np.argmax(data[:,2])
    max_r2_index = np.argmax(data[:,3])
    max_acc = np.array((data[max_acc_index,1],data[max_acc_index,4]))
    max_rms = np.array((data[max_acc_index,2],data[max_acc_index,5]))
    max_r2 = np.array((data[max_acc_index,3],data[max_acc_index,6]))

    return np.array((max_acc,max_rms,max_r2))

def max_class(data):
    max_acc_index = np.argmax(data[:,1])
    max_acc = np.array((data[max_acc_index,1],data[max_acc_index,2]))
    return max_acc

svm_acc = max_reg(svm_test_data)[0,:]
nn_reg_acc = max_reg(nn_reg_test_data)[0,:]
nn_class_acc = max_class(nn_class_test_data)

def wilson(data,conf):
    z=np.power(2,0.5)*erfinv(conf)
    wilson=np.zeros((np.size(data,axis=0),4))
    for count in range(0,np.size(data,axis=0)):
        n=np.floor(data[count,0]*7890)
        acc=data[count,1]

        pm = z/(1+z**2/n)*np.power(acc*(1-acc)/n+z**2/(4*n**2),0.5)
        x = (acc+z**2/(2*n))/(1+z**2/n)

        wilson[count,0],wilson[count,1],wilson[count,2],wilson[count,3] = data[count,0], x-pm,x+pm,2*pm
    return wilson

# def best_wilson(data):

print(wilson(svm_test_data,0.99)[4,:])
print(wilson(nn_reg_test_data,0.99)[4,:])
print(wilson(nn_class_test_data,0.99)[4,:])

