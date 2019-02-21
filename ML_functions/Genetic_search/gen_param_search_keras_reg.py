#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

def train_test_split(test_size,data):
#test_size = fraction of total data to be test sample
    length_train = int(np.size(data,axis=0)-np.floor(test_size*np.size(data,axis=0)))
    train_data=data[0:length_train,:]
    test_data=data[length_train:,:]
    return np.array((train_data,test_data))

def gen_y(NumPs,h11):
#return 1 if NumPs = h11 (favourable), 0 if not
    temp = np.zeros(np.size(h11))
    for count in range(0,np.size(h11)):
        if NumPs[count] == h11[count]:
            temp[count] = 1
        else:
            temp[count] = 0
    return temp

data=pd.read_csv('../../data',sep=",",skiprows=[0],header=None)
data=np.asfarray(data,float)
Numps = data[:,0]
h11 = data[:,2]

seed_0 = np.random.get_state()
np.random.seed(1)
np.random.shuffle(data)
np.random.set_state(seed_0)

train_data,test_data = train_test_split(0.70,data)

x_train,x_test = train_data[:,4:],test_data[:,4:]
y_train,y_test = gen_y(train_data[:,0],train_data[:,2]),gen_y(test_data[:,0],test_data[:,2])

def net_params(item):
    # no fully connected layers
    if item == 0:
        # return np.random.choice((1,2,3,4))
        return 1
    # fully connected sizes
    if item == 1:
        return int(np.floor(np.random.uniform(20,1000)))
    # if item == 2:
        # return int(np.floor(np.random.uniform(20,1000)))
    # if item == 3:
        # return int(np.floor(np.random.uniform(20,1000)))
    # if item == 4:
        # return int(np.floor(np.random.uniform(20,1000)))
    # if item == 0:
        # return np.random.uniform(0,10)
    # if item == 1:
        # return np.random.uniform(0.1,100)

from gen_param_functions_keras import create_population,evolve
from temp import evaluate_pop_scores

num_to_avg=5
variables=2
current_gen=create_population(20,variables,net_params)
np.savetxt('gen0',current_gen,fmt='%i ')
for gen in range(0,11):
    if gen == 0:
        scores,var = evaluate_pop_scores('train_keras_class',num_to_avg,x_train,y_train,x_test,y_test,current_gen,1,0,0,0)
    else:
        scores,var = evaluate_pop_scores('train_keras_class',num_to_avg,x_train,y_train,x_test,y_test,current_gen,0,prev_gen,prev_scores,prev_var)

    np.savetxt('acc'+str(gen),scores)
    np.savetxt('var'+str(gen),var)
    prev_gen, prev_scores, prev_var = current_gen, scores, var

    current_gen = evolve(current_gen,scores,variables,net_params)
    print(current_gen)
    np.savetxt('gen'+str(gen+1),current_gen,fmt='%i ')
