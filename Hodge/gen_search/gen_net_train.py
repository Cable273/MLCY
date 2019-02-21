#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Function Description:
#iterates over training data then back propogates to adjust weights

import numpy as np
import scipy as sp
# import matplotlib.pyplot as plt
import csv
import pandas as pd
import sys

def sgmd(x):
    return 1/(1+np.exp(-x))

def train(hidden_layers,hidden_layer_size,batch_size):
    hidden_layers=int(hidden_layers)
    batch_size=int(batch_size)
    hidden_layer_size=int(hidden_layer_size)

    first_hidden_size=0
    second_hidden_size=0
    if hidden_layers == 2:
        first_hidden_size = hidden_layer_size
        second_hidden_size = hidden_layer_size
    elif hidden_layers == 1:
        first_hidden_size = hidden_layer_size
        second_hidden_size = 0

    if second_hidden_size == 0:
        layer_sizes=np.array((180,first_hidden_size,1))
    else:
        layer_sizes=np.array((180,first_hidden_size,second_hidden_size,1))

    np.random.seed(2)
    # #Net parameters
    learn_rate=1
    momentum=0.9
    hidden_layers=int(np.size(layer_sizes)-2) #minus input output layers
    flat_spot_shift=0.01
    max_epoch=500

    #Import data
    data=pd.read_csv('../data',sep=",",skiprows=[0],header=None)
    data=np.asfarray(data,float)

    #shuffle input data 
    np.random.shuffle(data)

    input=data[:,4:]/5
    h11=data[:,2] 

    test_size=0.35
    length_train = int(np.size(input,axis=0)-np.floor(test_size*np.size(input,axis=0)))

    #transpose as thats how back prop is coded
    train_data=input[0:length_train,:].transpose()
    test_data=input[length_train+1:,:].transpose()

    train_size=np.size(train_data,axis=1) #no. of training data
    test_size=np.size(test_data,axis=1) #no. of test data
    input_size=np.size(train_data,axis=0) #no input neurons

    train_exp_out=h11[0:length_train]/19
    test_exp_out=h11[length_train+1:]/19

    # exp_out=np.zeros((102,int(np.size(input,axis=0))))
    # for count in range(0,train_size,1):
        # exp_out[int(h11[count]),count]=1

    # train_exp_out=exp_out[:,0:length_train]
    # test_exp_out=exp_out[:,length_train+1:]
    train_h11=h11[0:length_train]
    test_h11=h11[length_train+1:]

    #split input into batches to perform mini batch gradient descent
    input_batch=dict()
    exp_out_batch=dict()
    number_to_keep = np.floor(np.size(train_data,axis=1)/batch_size)*batch_size
    batches=int(number_to_keep/batch_size)
    lower=0
    for count in range(1,batches+1,1):
        input_batch[count]=train_data[:,lower:(count*batch_size)]
        # exp_out_batch[count]=train_exp_out[:,lower:(count*batch_size)]
        exp_out_batch[count]=train_exp_out[lower:(count*batch_size)]
        lower=lower+batch_size


    # weights=np.load('./trained_params/weights.npy').item()
    # biases=np.load('./trained_params/biases.npy').item()
    # error_array=np.load('./trained_params/errors.npy')
    # acc_array=np.load('./trained_params/accuracy.npy')

    #Create dictionary containing weight arrays for each layer
    weights = dict()
    #input -> first layer
    #distribution chosen for Xavier initialization
    weights[0]=np.random.normal(0,(1/layer_sizes[0])**2,(layer_sizes[1],layer_sizes[0])) 
    #hidden layers -> hidden layers
    if hidden_layers > 1:
        for count in range(1,hidden_layers,1):
            weights[count]=np.random.normal(0,(1/layer_sizes[0])**2,(layer_sizes[count+1],layer_sizes[count]))
    #last hidden layer -> output layer
    weights[hidden_layers]=np.random.normal(0,(1/layer_sizes[0])**2,(layer_sizes[int(np.size(layer_sizes)-1)],layer_sizes[int(np.size(layer_sizes)-2)]))

    #Create dictionary containing biases for each layer
    biases = dict()
    #input -> first layer
    biases[0]=np.random.uniform(-1,1,(layer_sizes[1]))
    #hidden layers -> hidden layers
    if hidden_layers > 1:
        for count in range(1,hidden_layers,1):
            biases[count]=np.random.uniform(-1,1,(layer_sizes[count+1]))
    #last hidden layer -> output layer
    biases[hidden_layers]=np.random.uniform(-1,1,(layer_sizes[int(np.size(layer_sizes)-1)]))

    #For diagnostics
    error_array=[]
    acc_array=[]
    test_acc_array=[]

    #dictionaries to contain output, delta shifts (back prop algorithm) and final shift arrays for each layer of network
    out=dict() 
    diag_out=dict()
    test_out=dict()
    deltas=dict() 
    weight_shifts=dict()
    bias_shifts=dict()

    #intialize shifts at zero for first run
    for count in range(0,hidden_layers+1,1):
        weight_shifts[count]=0
        bias_shifts[count]=0

    for epoch in range(1,max_epoch,1):
        for batch in range(1,batches+1,1):
            #Feed input through net for all trial data batch at a time
            out[0]=sgmd(np.dot(weights[0],input_batch[batch])+np.transpose(np.tile(biases[0],(batch_size,1)))) #hidden layer
            if hidden_layers > 1:
                for index in range(1,hidden_layers,1):
                    out[index]=sgmd(np.dot(weights[index],out[index-1])+np.transpose(np.tile(biases[index],(batch_size,1)))) #output layer with rescaling
            out[hidden_layers]=sgmd(np.dot(weights[hidden_layers],out[hidden_layers-1])+np.transpose(np.tile(biases[hidden_layers],(batch_size,1)))) #output layer with rescaling


            #back propogation
            #out*(1-out)+x to address flat spot problem
            deltas[hidden_layers]=(out[hidden_layers]-exp_out_batch[batch])*(out[hidden_layers]*(1-out[hidden_layers])+flat_spot_shift)
            for index in range(hidden_layers-1,-1,-1):
                deltas[index]=np.dot(np.transpose(weights[index+1]),deltas[index+1])*(out[index]*(1-out[index])+flat_spot_shift)

            #w_n_shift[u,v] ~ sum_over_all_training_data(delta^n(u) out^n-1(v))
            #equivalent of taking dot product of uth row of delta^n and vth row of out^n-1
            #ie matrix multiplication with one of them transposed
            #divide by batch size so learn rate is independant of sum over training data (shifts of the same order)
            for index in range(hidden_layers,0,-1):
                weight_shifts[index] = learn_rate / batch_size * np.dot(deltas[index],np.transpose(out[index-1])) + momentum * weight_shifts[index]
                bias_shifts[index] = learn_rate / batch_size * np.sum(deltas[index],axis=1) + momentum * bias_shifts[index]
            weight_shifts[0] = learn_rate / batch_size * np.dot(deltas[0],np.transpose(input_batch[batch])) + momentum * weight_shifts[0]
            bias_shifts[0] = learn_rate / batch_size * np.sum(deltas[0],axis=1) + momentum * bias_shifts[0]

            for index in range(0,hidden_layers+1,1):
                weights[index] = weights[index] - weight_shifts[index]
                biases[index] = biases[index] - bias_shifts[index]

        ##diagnostics, error func and accuracy##
        # Feed input through net for all trial data for error+accuracy measurements
        diag_out[0]=sgmd(np.dot(weights[0],train_data)+np.transpose(np.tile(biases[0],(train_size,1)))) #hidden layer
        if hidden_layers > 1:
            for index in range(1,hidden_layers,1):
                diag_out[index]=sgmd(np.dot(weights[index],diag_out[index-1])+np.transpose(np.tile(biases[index],(train_size,1)))) #output layer with rescaling
        diag_out[hidden_layers]=sgmd(np.dot(weights[hidden_layers],diag_out[hidden_layers-1])+np.transpose(np.tile(biases[hidden_layers],(train_size,1)))) #output layer with rescaling

        # check training data against net for accuracy
        correct=0
        for index in range(0,train_size,1):
            if np.round(19*diag_out[hidden_layers][0,index]) == train_h11[index]:
                correct=correct+1
        accuracy=correct/train_size

        # correct=0
        # for index in range(0,train_size,1):
            # entry=int(train_h11[index])
            # if np.round(diag_out[hidden_layers][entry,index])==1:
                # correct=correct+1
        # accuracy=correct/train_size
        # #check acc of test data

    print(accuracy)
    return accuracy
