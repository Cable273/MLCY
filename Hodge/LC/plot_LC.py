#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

svm_train_data = np.loadtxt('./hodge_svm_train_LC_avg')
svm_test_data = np.loadtxt('./hodge_svm_test_LC_avg')

nn_reg_train_data = np.loadtxt('./hodge_nn_reg_train_LC_avg')
nn_reg_test_data = np.loadtxt('./hodge_nn_reg_test_LC_avg')

nn_class_train_data = np.loadtxt('./hodge_nn_class_train_LC_avg')
nn_class_test_data = np.loadtxt('./hodge_nn_class_test_LC_avg')

x = 1 - svm_train_data[:,0]
print(nn_class_test_data)

marker_size = 4
cap_size = 3.5

plt.errorbar(x, svm_test_data[:,1], yerr=svm_test_data[:,4], fmt='o', markersize=marker_size, capsize=cap_size,label='SVM Regressor Validation Accuracy',color = 'purple')
plt.plot(x, svm_test_data[:,1],color='purple')

plt.errorbar(x, nn_reg_test_data[:,1], yerr=nn_reg_test_data[:,4], fmt='o', markersize=marker_size, capsize=cap_size,label='Neural Net Regressor Validation Accuracy',color = 'green')
plt.plot(x, nn_reg_test_data[:,1],color='green')

plt.errorbar(x, nn_class_test_data[:,1], yerr=nn_class_test_data[:,2], fmt='o', markersize=marker_size, capsize=cap_size,label='Neural Net Classifier Validation Accuracy',color = 'orange')
plt.plot(x, nn_class_test_data[:,1],color='orange')

plt.xlabel('Fraction of data used for training')
plt.ylabel('Accuracy')
plt.title('Hodge Number - Validation Learning Curves')
plt.legend()
plt.savefig('Hodge_test_LC.pdf')
plt.show()

# plt.errorbar(x, svm_train_data[:,1], yerr=svm_train_data[:,4], fmt='o', markersize=marker_size, capsize=cap_size,label='SVM Classifier Training Accuracy',color = 'purple')
# plt.plot(x, svm_train_data[:,1],color='purple')

# plt.errorbar(x, nn_reg_train_data[:,1], yerr=nn_reg_train_data[:,4], fmt='o', markersize=marker_size, capsize=cap_size,label='Neural Net Regressor, Validation Accuracy',color = 'green')
# plt.plot(x, nn_reg_train_data[:,1],color='green')

# plt.errorbar(x, nn_class_train_data[:,1], yerr=nn_class_train_data[:,2], fmt='o', markersize=marker_size, capsize=cap_size,label='Neural Net Classifier, Training Accuracy',color = 'orange')
# plt.plot(x, nn_class_train_data[:,1],color='orange')

# plt.xlabel('Fraction of data used for training')
# plt.ylabel('Accuracy')
# plt.title('Hodge Number - Training Learning Curves')
# plt.legend()
# # plt.savefig('Hodge_train_LC.pdf')
# plt.show()
