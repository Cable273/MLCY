#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

def train_keras_class(x_train,y_train,x_test,y_test,network_params):
    import keras
    from keras.models import Sequential,load_model
    from keras.layers import Dense,Activation,Dropout,Conv1D,Conv2D,MaxPooling1D,MaxPooling2D,Flatten
    from keras.callbacks import EarlyStopping

    dropout = network_params[0]
    no_full_connect_layers = int(network_params[1])
    layer_sizes = network_params[2:].astype(int)

    network = Sequential()
    #Initial layer
    network.add(Dense(layer_sizes[0],input_dim=np.size(x_train,axis=1)))
    network.add(Activation('relu'))
    network.add(Dropout(dropout))
    #Hidden layers
    for count in range(1,no_full_connect_layers):
        network.add(Dense(layer_sizes[count]))
        network.add(Activation('relu'))
        network.add(Dropout(dropout))
    #Output layer
    network.add(Dense(1))
    network.add(Activation('sigmoid'))

    # val_acc_stop = EarlyStopping(monitor='val_acc',min_delta=0,patience=5,verbose=0,mode='auto')
    val_loss_stop = EarlyStopping(monitor='val_loss',min_delta=0,patience=5,verbose=0,mode='auto')
    network.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
    # network.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
    history = network.fit(x_train,y_train,batch_size=32,epochs=1000,verbose=0,shuffle=False,validation_data=(x_test,y_test),callbacks=[val_loss_stop])

    # def class_to_hodge(y):
        # h11 = np.zeros(np.size(y,axis=0))
        # for count in range(0,np.size(y,axis=0)):
            # h11[count] = np.argmax(y[count,:])
        # return h11
    # train_predict = class_to_hodge(network.predict(x_train,verbose = 0))
    # test_predict = class_to_hodge(network.predict(x_test,verbose = 0))
    # y_train,y_test = class_to_hodge(y_train),class_to_hodge(y_test)

    # train_predict = network.predict(x_train,verbose = 0)
    # test_predict = network.predict(x_test,verbose = 0)

    def sgmd_thresh(x,threshold):
        return 1/(1+np.exp(-(x-threshold)))

    def inv_sgmd(x):
        return -np.log(1/x - 1)

    train_predict_orig = inv_sgmd(network.predict(x_train,verbose = 0)[:,0])
    test_predict_orig = inv_sgmd(network.predict(x_test,verbose = 0)[:,0])
    # dec_bound=np.arange(-100,np.max(test_predict_orig),1)
    dec_bound=np.arange(-100,120,1)

    def max_min_inf(F):
        max_F=-1000
        min_F=1000
        for count in range(0,np.size(F,axis=0)):
            if F[count] > max_F:
                if F[count] != np.inf:
                    if F[count] != -np.inf:
                        max_F = F[count]
            if F[count] < min_F:
                if F[count] != np.inf:
                    if F[count] != -np.inf:
                        min_F = F[count]
        return np.array((max_F,min_F))
    max_dec,min_dec = max_min_inf(test_predict_orig)
    # print(np.max(test_predict_orig))
    # print(min_dec,max_dec)
    # dec_bound=np.arange(min_dec,max_dec,1)

    F_test_vals = np.zeros(np.size(dec_bound))
    for count in range(0,np.size(dec_bound,axis=0)):
        train_predict = sgmd_thresh(train_predict_orig,dec_bound[count])
        test_predict = sgmd_thresh(test_predict_orig,dec_bound[count])
        
        from performance_metrics import acc_metrics,ROC,F,class_acc
        F_test_vals[count]=F(test_predict,y_test,'nn')
    # plt.plot(F_test_vals)
    # plt.show()

    def max_min_nan(F):
        max_F=0
        for count in range(0,np.size(F,axis=0)):
            if F[count] != 'nan':
                if F[count] > max_F:
                    max_F = F[count]
        return max_F
    F_test=max_min_nan(F_test_vals)

    # from performance_metrics import acc_metrics,ROC,F,class_acc,reg_acc
    # metrics_train,metrics_test = acc_metrics(train_predict,y_train,'nn'),acc_metrics(test_predict,y_test,'nn')
    # ROC_train, ROC_test = ROC(train_predict,y_train,'nn'),ROC(test_predict,y_test,'nn')
    # F_train, F_test = F(train_predict,y_train,'nn'),F(test_predict,y_test,'nn')
    # # acc_train, acc_test = class_acc(train_predict,y_train,'nn'),class_acc(test_predict,y_test,'nn')
    # acc_train, acc_test = reg_acc(train_predict,y_train,0.1),reg_acc(test_predict,y_test,0.1)

    # return np.array((acc_test,F_test))
    return np.array((F_test,2))

def train_keras_reg(x_train,y_train,x_test,y_test,network_params):
    import keras
    from keras.models import Sequential,load_model
    from keras.layers import Dense,Activation,Dropout,Conv1D,Conv2D,MaxPooling1D,MaxPooling2D,Flatten
    from keras.callbacks import EarlyStopping

    dropout = network_params[0]
    no_full_connect_layers = int(network_params[1])
    layer_sizes = network_params[2:].astype(int)

    #normalize outputs bu max
    n=np.max(np.array((np.max(y_train),np.max(y_test))))
    y_train,y_test = y_train/n, y_test/n

    network = Sequential()
    #Initial layer
    network.add(Dense(layer_sizes[0],input_dim=np.size(x_train,axis=1)))
    network.add(Activation('relu'))
    network.add(Dropout(dropout))
    #Hidden layers
    for count in range(1,no_full_connect_layers):
        network.add(Dense(layer_sizes[count]))
        network.add(Activation('relu'))
        network.add(Dropout(dropout))
    #Output layer
    network.add(Dense(1))

    # val_acc_stop = EarlyStopping(monitor='val_acc',min_delta=0,patience=25,verbose=1,mode='auto')
    val_loss_stop = EarlyStopping(monitor='val_loss',min_delta=0,patience=20,verbose=0,mode='auto')
    network.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
    history = network.fit(x_train,y_train,batch_size=32,epochs=1000,verbose=0,shuffle=False,validation_data=(x_test,y_test),callbacks=[val_loss_stop])

    train_predict = network.predict(x_train,verbose = 0)[:,0]
    test_predict = network.predict(x_test,verbose = 0)[:,0]

    from performance_metrics import rms_error,reg_acc,r2
    rms_train,rms_test = rms_error(n*train_predict,n*y_train),rms_error(n*test_predict,n*y_test)
    acc_train,acc_test = reg_acc(n*train_predict,n*y_train,0.5),reg_acc(n*test_predict,n*y_test,0.5)
    r2_train,r2_test = r2(n*train_predict,n*y_train),r2(n*test_predict,n*y_test)
    # print(rms_train,rms_test)
    # print(acc_train,acc_test)
    # print(r2_train,r2_test)

    return np.array((acc_test,rms_test,r2_test))

def train_svm_reg(x_train,y_train,x_test,y_test,kernel,ker_param,slack,res):
    from svm_functions import svm_dual_reg,gen_model_reg,bulk_svm_reg_eval
    from svm_kernels import lin_kernel,poly_kernel,gauss_kernel

    from cvxopt import solvers
    solvers.options['show_progress'] = False
    alphas = svm_dual_reg(x_train,y_train,kernel,ker_param,slack,res)
    model = gen_model_reg(alphas,x_train)

    #eval regressor with no shift
    train_predict = bulk_svm_reg_eval(x_train,model,kernel,ker_param,0)
    test_predict = bulk_svm_reg_eval(x_test,model,kernel,ker_param,0)

    #find shifts from average distance
    b_train = np.mean(y_train - train_predict)
    b_test = np.mean(y_test - test_predict)

    #eval regressor with correct shift
    train_predict = bulk_svm_reg_eval(x_train,model,kernel,ker_param,b_train)
    test_predict = bulk_svm_reg_eval(x_test,model,kernel,ker_param,b_test)

    from performance_metrics import rms_error,reg_acc,r2
    rms_train,rms_test = rms_error(train_predict,y_train),rms_error(test_predict,y_test)
    acc_train,acc_test = reg_acc(train_predict,y_train),reg_acc(test_predict,y_test)
    r2_train,r2_test = r2(train_predict,y_train),r2(test_predict,y_test)
    # print(rms_train,rms_test)
    # print(acc_train,acc_test)
    # print(r2_train,r2_test)

    return acc_test

def train_svm_class(x_train,y_train,x_test,y_test,kernel,ker_param,slack):
    from svm_functions import svm_dual,gen_model,b_from_model,bulk_svm_eval,svm_eval
    from svm_kernels import lin_kernel,poly_kernel,gauss_kernel

    from cvxopt import solvers
    solvers.options['show_progress'] = False
    alphas = svm_dual(x_train,y_train,kernel,ker_param,slack)
    model = gen_model(alphas,x_train,y_train)
    b = b_from_model(model,kernel,ker_param)

    train_predict = bulk_svm_eval(x_train,model,kernel,ker_param,b)
    test_predict = bulk_svm_eval(x_test,model,kernel,ker_param,b)

    from performance_metrics import acc_metrics,ROC,F,class_acc

    metrics_train,metrics_test = acc_metrics(train_predict,y_train,'svm'),acc_metrics(test_predict,y_test,'svm')
    ROC_train, ROC_test = ROC(train_predict,y_train,'svm'),ROC(test_predict,y_test,'svm')
    F_train, F_test = F(train_predict,y_train,'svm'),F(test_predict,y_test,'svm')
    acc_train, acc_test = class_acc(train_predict,y_train,'svm'),class_acc(test_predict,y_test,'svm')

    # print(metrics_train)
    # print(metrics_test)
    # print(ROC_train)
    # print(ROC_test)
    # print(F_train,F_test)
    # print(acc_train,acc_test)

    return np.array((acc_test,F_test))
