#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Function Description:
#iterates over training data then back propogates to adjust weights

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(1) #must be before keras

import tensorflow as tf
import keras
from keras.models import Sequential
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

def train_keras_class(network_params):
    # no_conv_layers = int(network_params[0])
    # conv_kernel_size = int(network_params[1])
    # conv_filters = network_params[2:6].astype(int)
    no_full_connect_layers = int(network_params[0])
    layer_sizes = network_params[1:].astype(int)

    #Import data
    data=pd.read_csv('../../data',sep=",",skiprows=[0],header=None)
    data=np.asfarray(data,float)

    np.random.shuffle(data)

    inputs=data[:,4:]/5
    h11=data[:,2] 
    h12=data[:,3] 

    train_size=np.size(inputs,axis=0)

    test_size=0.1267427
    length_train = int(np.size(inputs,axis=0)-np.floor(test_size*np.size(inputs,axis=0)))

    train_data=inputs[0:length_train,:]
    test_data=inputs[length_train:,:]

    # train_exp_out=h11[0:length_train]
    # test_exp_out=h11[length_train+1:]

    exp_out=np.zeros((int(np.size(inputs,axis=0)),20))
    for count in range(0,train_size,):
        exp_out[count,int(h11[count])]=1
    train_exp_out=exp_out[0:length_train,:]
    test_exp_out=exp_out[length_train:,:]

    # train_data=np.reshape(train_data,(length_train,12,15,1))
    # test_data=np.reshape(test_data,(int(np.size(inputs,axis=0)-length_train),12,15,1))

    #Create network architechture
    network = Sequential()
    # Convolution layers
    # network.add(Conv2D(conv_filters[0],kernel_size=(conv_kernel_size,conv_kernel_size),input_shape=(12,15,1)))
    # network.add(Activation('relu'))
    # for count in range(0,no_conv_layers-1):
        # network.add(Conv2D(conv_filters[count+1],kernel_size=(conv_kernel_size,conv_kernel_size)))
        # network.add(Activation('relu'))

    #Fully connected layers
   # network.add(Flatten())
    network.add(Dense(layer_sizes[0],input_dim=180))
    for count in range(1,no_full_connect_layers):
        network.add(Dense(layer_sizes[count]))
        network.add(Activation('relu'))
        network.add(Dropout(0.5))

    #output layer
    network.add(Dense(20))
    network.add(Activation('softmax'))

    val_acc_stop = EarlyStopping(monitor='val_acc',min_delta=0,patience=5,verbose=0,mode='auto')
    val_loss_stop = EarlyStopping(monitor='val_loss',min_delta=0,patience=5,verbose=0,mode='auto')

    network.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    history = network.fit(train_data,train_exp_out,batch_size=32,epochs=15050,verbose=0,shuffle=False,validation_data=(test_data,test_exp_out),callbacks=[val_acc_stop,val_loss_stop])

    val_acc=np.array(history.history['val_acc'])
    final_val_accuracy = val_acc[np.size(val_acc)-1]

    return final_val_accuracy
