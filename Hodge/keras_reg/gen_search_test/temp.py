#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

#check populations to see if network already in it
#returns array, [0 or 1,index], 1 is true (exists in population)
# index is which entry of prev population it is
def check_existing_pop(network,check_pop):
    in_pop=0
    pop_index=0
    for count in range(0,np.size(check_pop,axis=0)):
        if np.array_equal(check_pop[count,:],network):
            in_pop = 1
            pop_index = count
            break
    return np.array((in_pop,pop_index))

from ml_training_variable import train_svm_class,train_svm_reg,train_keras_class,train_keras_reg
#Evaluate average accuracy and variance of accuracy for population of networks
#set is_inital=1 for first gen, prev variables then dummy var
def evaluate_pop_scores(training_method,num_to_avg,x_train,y_train,x_test,y_test,current_gen,is_initial,prev_gen,prev_scores,prev_var):
    scores=np.zeros(np.size(current_gen,axis=0))
    var=np.zeros(np.size(current_gen,axis=0))
    for network in range(0,np.size(current_gen,axis=0)):
        #for gen0, no need to check if in previous gen
        if is_initial == 1:
            temp=np.zeros(num_to_avg)
            for count in range(0,num_to_avg):
                # temp[count]=globals()[training_method](x_train,y_train,x_test,y_test,'gauss_kernel',current_gen[network,0],current_gen[network,1])[0]
                temp[count]=globals()[training_method](x_train,y_train,x_test,y_test,current_gen[network,:])[0]
            scores[network] = np.mean(temp)
            var[network] = np.var(temp)

        #if exists in previous or current gen, skip training and take that score
        elif is_initial == 0:
            in_prev_gen, prev_gen_index = check_existing_pop(current_gen[network,:],prev_gen)
            in_current_gen, existing_current_index = check_existing_pop(current_gen[network,:],current_gen[0:network,:])

            if in_prev_gen == 1:
                scores[network] = prev_scores[prev_gen_index]
                var[network] = prev_var[prev_gen_index]
            elif in_current_gen ==1:
                scores[network] = scores[existing_current_index]
                var[network] = var[existing_current_index]

            else:
                temp=np.zeros(num_to_avg)
                for count in range(0,num_to_avg):
                    # temp[count] = globals()[training_method](x_train,y_train,x_test,y_test,'gauss_kernel',current_gen[network,0],current_gen[network,1])[0]
                    temp[count]=globals()[training_method](x_train,y_train,x_test,y_test,current_gen[network,:])[0]
                scores[network] = np.mean(temp)
                var[network] = np.var(temp)
        print(network,scores[network],var[network])

    return np.array((scores,var))
