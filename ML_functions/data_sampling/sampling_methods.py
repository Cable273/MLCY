#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

def count_true(y):
    true_count = 0
    for count in range(0,np.size(y,axis = 0)):
        if np.round(y[count]) == 1:
            true_count += 1
    return true_count

def true_false_split(x,y,tf):
    nt = count_true(y)
    nf = np.size(y)-nt

    x_true, x_false = np.zeros(np.size(x,axis = 1)), np.zeros(np.size(x,axis = 1))
    for count in range(0,np.size(x,axis = 0)):
        if y[count] == 1:
            x_true = np.vstack((x_true,x[count,:]))
        else:
            x_false = np.vstack((x_false,x[count,:]))
    x_true, x_false = np.delete(x_true,0,axis=0), np.delete(x_false,0,axis=0)
    if tf == 'True':
        return x_true
    elif tf == 'False':
        return x_false

def downsample(x,y,N):
#downsample data, nf -> scale*nf,so frac of true outputs = frac
    nt = count_true(y)
    nf = np.size(y)-nt
    x_true,x_false = true_false_split(x,y,'True'),true_false_split(x,y,'False')

    # scale = nt/nf*(1/frac-1)
    scale = nt/nf*100/N
    len_false = int(scale*nf)

    #shuffle and keep data
    np.random.shuffle(x_false)
    x_downsampled = np.vstack((x_true,x_false[0:len_false]))
    np.random.shuffle(x_downsampled)
    return x_downsampled

def upsample(x,y,frac):
#upsample data, nt->scale*nt, so frac of true outputs = frac
    nt = count_true(y)
    nf = np.size(y)-nt
    x_true,x_false = true_false_split(x,y,'True'),true_false_split(x,y,'False')

    scale = nf/nt*(frac/(1-frac))
    len_true = int(scale*nt)

    # N = int(np.floor((np.size(x_false,axis=0)-1)/np.size(x_true,axis=0)))
    N = int(np.floor(scale))
    temp = np.zeros((N*np.size(x_true,axis=0),np.size(x_true,axis=1)))
    for n in range(0,N):
        L = np.size(x_true,axis=0)
        temp[n*L:(n+1)*L,:]=x_true

    x_upsampled = np.vstack((temp,x_false))
    np.random.shuffle(x_upsampled)
    return x_upsampled

from smote import nearest_neighbours,gen_synth_point,smote
def smote_sample(x,y,N):
    nt = count_true(y)
    nf = np.size(y)-nt
    # N = int(np.floor((nf/nt*frac/(1-frac)-1)*100))
    x_true,x_false = true_false_split(x,y,'True'),true_false_split(x,y,'False')
    
    #generate synthetic true values
    smoted_data = smote(x_true,5,N)
    #fix column of lowest sym_order to 1 for these, so they are classed as having a sym
    smoted_data[:,2] = np.ones(np.size(smoted_data,axis=0))

    smote_sampled = np.vstack((x_true,smoted_data))
    smote_sampled = np.vstack((x_false,smote_sampled))
    np.random.shuffle(smote_sampled)
    return smote_sampled
