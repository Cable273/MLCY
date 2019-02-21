#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

poly_c = 1
#Kernels: Finds matrix M_ij = ker(x_i,y_j)
#input X,Y of the form X[i,j],i vectors, j dimension of each vector
# if x,y single vectors, returns scalar of single kernel product

def lin_kernel(x,y,dummy):
#linear, kernel(x_i,y_j)=np.dot(x_i,y_j)
#sum_u(x^i_u*y^j_u) = X_{iu}*Y{ju}=(X*Y^T)_{ij}
    return np.dot(x,y.T)

def gauss_kernel(x,y,std):
#Gaussian, kernel(x_i,x_j)=exp(|x_i-x_j|^2/(2*std^2))
#|x_i-y_j|^2 = x^i_u*x^i_u + y^j_u*y^j_u - 2*x^i_u*y^j_u
#""= (X*X^T)_{ii} + (Y*Y^T)_{jj} - 2*(X*Y^T)_{ij}
#represent (X*X^T)_{ii} as vector and use broadcasting for sum
    XXT,YYT,XYT = np.dot(x,x.T),np.dot(y,y.T),np.dot(x,y.T)
    if np.size(XXT) != 1 and np.size(YYT) != 1:
        #Take diagonal of XXT_{ii} to be vector
        XXT_i,YYT_i = np.diag(XXT),np.diag(YYT)
        #Broadcasting to do "tensor sum" V_ij = xxt_i+yyt_j
        Vij = XXT_i.reshape(-1,1)+YYT_i
        #Finally evaluate difference squared for every vector in x,y
        xy_ij_squared = Vij -2*XYT
        return np.exp(-xy_ij_squared/(2*np.power(std,2)))
    elif np.size(XXT) == 1 and np.size(YYT) != 1:
        YYT_i = np.diag(YYT)
        Yx_i = np.dot(y,x)
        xy_i_squared = XXT + YYT_i - 2*Yx_i
        return np.exp(-xy_i_squared/(2*np.power(std,2)))
    elif np.size(XXT) != 1 and np.size(YYT) == 1:
        XXT_i = np.diag(XXT)
        Xy_i = np.dot(x,y)
        xy_i_squared = YYT + XXT_i - 2*Xy_i
        return np.exp(-xy_i_squared/(2*np.power(std,2)))
    else:
        return np.exp(-np.dot(x-y,x-y)/(2*np.power(std,2)))

def poly_kernel(x,y,n):
#Poly, k(x_i,y_j) = (1+dot(x,y))^n
    return np.power(poly_c+np.dot(x,y.T),n)
