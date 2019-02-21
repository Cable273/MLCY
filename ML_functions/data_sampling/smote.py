#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

def nearest_neighbours(x,x_set,k):
    #calculate distance of every point in x_set from x
    #d_j = (x-x_j)^2 
    d=np.power(np.sum((x-x_set)**2,axis=1),0.5)

    #obtain k nn
    nn = np.zeros((k,np.size(x)))
    i = 0
    while i < k:
        lowest_index = np.argmin(d)

        #discard zero, means x is in sample
        if d[lowest_index] == 0:
            d = np.delete(d,lowest_index,axis=0)
            x_set = np.delete(x_set,lowest_index,axis=0)
        else:
            nn[i,:] = x_set[lowest_index,:]
            i += 1
            d = np.delete(d,lowest_index,axis=0)
            x_set = np.delete(x_set,lowest_index,axis=0)
    return nn

def gen_synth_point(x,x_set,k):
#gen synthetic duplicates by scaling difference vectors of nearest n and adding to x
    nn = nearest_neighbours(x,x_set,k)

    #calculate array of difference vector for every nn
    d_vec = nn - x

    #scale each d_vec by random no. between [0,1]
    for count in range(0,np.size(d_vec,axis =0)):
        d_vec[count,:] = np.random.uniform(0,1)*d_vec[count,:]

    #calculate new synthetic points r=r+d
    synth = x + d_vec
    return synth

def smote(x,k,N):
    n = int(N/100)
    generated_samples = np.zeros((np.size(x,axis = 0)*n,np.size(x,axis=1)))
    #sweep samples, generate k nn for each sample
    for sample in range(0,np.size(x,axis=0)):
        for j in range(0,n):
            synth = gen_synth_point(x[sample,:],x,k)
            #pick just one k out of sample
            index = np.random.choice(np.arange(0,k))
            generated_samples[sample*n+j,:] = synth[index,:]

    return generated_samples
