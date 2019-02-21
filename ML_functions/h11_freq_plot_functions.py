#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

def count_bins(y,y_exp):
    y=np.round(y)
    poss_vals=np.arange(0,np.max(y_exp))
    value_freq=np.zeros(np.size(poss_vals))

    for val_index in range(0,np.size(poss_vals,axis=0)):
        for count in range(0,np.size(y,axis=0)):
            if y[count] == poss_vals[val_index]:
                value_freq[val_index] += 1
    return value_freq

def plot_h11_freq(y,y_exp,split):
    #The data
    pred_bin_count = count_bins(y,y_exp)
    act_bin_count = count_bins(y_exp,y_exp)
    bin_index=np.arange(0,np.max(y_exp))

    #export data
    M=np.transpose(np.array((bin_index,act_bin_count,pred_bin_count)))
    np.savetxt('h11_freq_data.csv',M,fmt='%i',delimiter=',')

    #Calculate optimal width
    width = np.min(np.diff(bin_index))/3

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(bin_index-width,act_bin_count,width,color='g',label='True Hodge Data')
    ax.bar(bin_index,pred_bin_count,width,color='y',label='Model Prediction')
    plt.xticks(bin_index)
    plt.legend()
    plt.xlabel('Hodge number h_11')
    plt.ylabel('Frequency')
    plt.title('h11 frequency of validation set ('+str(100*split)+'% of full dataset)')
    plt.savefig('h11_freq.pdf')
    plt.show()
