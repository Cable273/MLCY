#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

def acc_metrics(y,y_exp,learner_type):
    tp,tn,fp,fn = 0,0,0,0
    for count in range(0,np.size(y,axis = 0)):
        if y_exp[count] == 1:
            if np.round(y[count]) == y_exp[count]:
                tp +=1
            else:
                fn +=1
        if learner_type == 'svm': #svm have +-1 output
            if y_exp[count] == -1:
                if np.round(y[count]) == y_exp[count]:
                    tn +=1
                else:
                    fp +=1
        elif learner_type == 'nn': #NN use 0/1 output
            if y_exp[count] == 0:
                if np.round(y[count]) == y_exp[count]:
                    tn +=1
                else:
                    fp +=1
    return np.array((tp,tn,fp,fn))

#for use with "classfier" when used as a regressor
def acc_metrics_reg(y,y_exp,learner_type):
    tp,tn,fp,fn = 0,0,0,0
    for count in range(0,np.size(y,axis = 0)):
        if learner_type == 'svm': #svm have +-1 output
            if y_exp[count] == -1:
                if np.round(y[count]) == y_exp[count]:
                    tn +=1
                else:
                    fp +=1
            if y_exp[count] != -1:
                if np.round(y[count]) == y_exp[count]:
                    tp +=1
                else:
                    fn +=1
        elif learner_type == 'nn': #NN use 0/1 output
            if y_exp[count] == 0:
                if np.round(y[count]) == y_exp[count]:
                    tn +=1
                else:
                    fp +=1
            if y_exp[count] != 0:
                if np.round(y[count]) == y_exp[count]:
                    tp +=1
                else:
                    fn +=1
    return np.array((tp,tn,fp,fn))

def ROC(y,y_exp,learner_type):
    tp,tn,fp,fn=acc_metrics(y,y_exp,learner_type)
    TP_rate = tp/(tp+fn)
    FP_rate = fp/(fp+tn)
    return(TP_rate,FP_rate)

def F(y,y_exp,learner_type):
    tp,tn,fp,fn=acc_metrics(y,y_exp,learner_type)
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    return 2/(1/recall+1/precision)

def F_reg(y,y_exp,learner_type):
    tp,tn,fp,fn=acc_metrics_reg(y,y_exp,learner_type)
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    return 2/(1/recall+1/precision)

def class_acc(y,y_exp,learner_type):
    tp,tn,fp,fn=acc_metrics(y,y_exp,learner_type)
    print(tp,tn,fp,fn)
    return (tp+tn)/(tp+tn+fp+fn)

def rms_error(y,y_exp):
    N = np.size(y,axis=0)
    rms = np.sum(np.absolute(y-y_exp))/N
    return rms

def reg_acc(y,y_exp,tol):
    correct = 0
    for count in range(0,np.size(y)):
        if y[count]-tol <  y_exp[count] < y[count]+tol:
            correct+=1
    return correct/np.size(y)

def r2(y,y_exp):
    SS_res = np.sum((y-y_exp)**2)
    SS_tot = np.sum((y_exp-np.mean(y_exp))**2)
    return 1-SS_res/SS_tot

#matthews phi, for binary classifiers only
def MCC(y,y_exp,learner_type):
    tp,tn,fp,fn=acc_metrics(y,y_exp,learner_type)
    MCC = (tp*tn-fp*fn)*np.power((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn),-0.5)
    return MCC
