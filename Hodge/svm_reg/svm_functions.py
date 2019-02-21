#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from cvxopt import matrix,solvers
from sklearn import svm, datasets

from svm_kernels import lin_kernel,gauss_kernel,poly_kernel

#ONLY POSSIBLE WITH LINEARLY SEP DATA. USE DUAL AND KERNELS FOR NON LINEAR
#find hyperplane of the form dot(r,w)+b=0. Returns w and b values
#solves using cvxopt, Min 1/2*w^T*P*w+q*w subject to G*w<=h,A*w=b
def svm_primal(x,y):
#Inputs: x=points in parameter space, y=classification (+-1)
#SVM_primal:Min w^2  with constraints y_i(w*x_i+b)>=1

    #Treat w=(w_1,w_2,...,b), hence w^2 P=diag(1,1,...,0)
    P=np.zeros((np.size(x,axis=1)+1,np.size(x,axis=1)+1))
    np.fill_diagonal(P,1)
    P[np.size(x,axis=1),np.size(x,axis=1)] = 0

    q = np.zeros(np.size(x,axis=1)+1,dtype=float)

    G=np.zeros((np.size(x,axis=0),3))
    for count in range(0,np.size(x,axis=0)):
        G[count,0] = - y[count] * x[count,0]
        G[count,1] = - y[count] * x[count,1]
        G[count,2] = - y[count] 
    h = -np.ones(np.size(x,axis=0),dtype=float) 

    P,q,G,h = matrix(P), matrix(q), matrix(G), matrix(h)

    sol=solvers.qp(P,q,G,h)
    params=sol['x']
    return params

def svm_dual(x,y,kernel,ker_param,slack):
#Inputs: x=points in parameter space, y=classification (+-1)
#Inputs: kernel name of  kernel used. ker_param std for gaus, poly degree for poly
#cvxopt: min 1/2 x^T*P*x - q^T x subject to Gx<=h,Ax=b

#SVM_dual:min 1/2*sum_(ij)[a_i*a_j*y_i_y_j*dot(x_i,x_j)]-sum_i[a_i]
#SVM_dual:constraints a_i>0, sum_i[y_i*a_i] = 0
#SVM_dual:if slack != 0, constraint 0<a_i<slack

    #P_ij = y_i*y_j*Kernel(x_i,x_j), kernel pairwise efficient matrix implementation for given calculating all pairs x_i,x_j
    # P=np.multiply(np.tensordot(y,y,0),kernel_pairwise(x,x))
    P=np.multiply(np.tensordot(y,y,0),globals()[kernel](x,x,ker_param))

    q=-np.ones(np.size(x,axis=0))

    #a_i>0
    G=-np.eye(np.size(x,axis=0))
    h=np.zeros(np.size(x,axis=0))

    if slack != 0:
        G_slack = np.eye(np.size(x,axis=0))
        h_slack = slack*np.ones(np.size(x,axis=0))
        G=np.vstack((G,G_slack))
        h=np.append(h,h_slack)

    #sum_i[a_i*y_i]=0
    A=np.array((y),dtype=float)
    A=matrix(A,(1,np.size(x,axis=0)))
    b=matrix(0.0)

    P,q,G,h = matrix(P), matrix(q), matrix(G), matrix(h)

    # solvers.options['maxiters']=1200
    # solvers.options['show_progress']=False
    sol=solvers.qp(P,q,G,h,A,b)
    params=sol['x']
    return np.array((params)).transpose()[0]

def svm_dual_reg(x,y,kernel,ker_param,slack,res):
#Inputs: x=points in parameter space, y=output
#Inputs: kernel name of  kernel used. ker_param std for gaus, poly degree for poly
#cvxopt: min 1/2 x^T*P*x - q^T x subject to Gx<=h,Ax=b

#SVM_dual:min 1/2*sum_(ij)[(a_i-a_i*)(a_j-a_j*)ker(x_i,x_j)+res*sum_i(ai+ai*)-sum_i(yi(ai-ai*))

#SVM_dual:constraints a_i,a_i*>0, sum_i[ai-ai*]=0
#SVM_dual:if slack != 0, constraint 0<a_i<slack

    #generate P matrix in quadrants
    #P_ij = P_{i+n,j+n} = ker_ij,P_{i+n,j}=P_{i,j+n}=-ker_{ij}
    ker_ij=globals()[kernel](x,x,ker_param)
    N = np.size(x,axis =0)
    P = np.zeros((2*N,2*N))
    P[:N,:N],P[N:,N:] = ker_ij,ker_ij
    P[N:,:N],P[:N,N:] = -ker_ij,-ker_ij

    q = res * np.ones(2*N) - np.append(y,-y)

    #a_i>0
    G=-np.eye(2*N)
    h=np.zeros(2*N)

    if slack != 0:
        G_slack = np.eye(2*N)
        h_slack = slack*np.ones(2*N)
        G=np.vstack((G,G_slack))
        h=np.append(h,h_slack)

    #sum_i[a_i*y_i]=0
    A=np.append(np.ones(N),-np.ones(N))
    A=matrix(A,(1,2*N))
    b=matrix(0.0)

    P,q,G,h = matrix(P), matrix(q), matrix(G), matrix(h)

    solvers.options['maxiters']=1200
    # solvers.options['show_progress']=False
    sol=solvers.qp(P,q,G,h,A,b)
    params=sol['x']
    return np.array((params)).transpose()[0]

def gen_model(alphas,x,y):
#Returns array with each row (alpha,vector_ouput,support vector)
#ie everything needed to construct hyperplane, x_i,y_i,a_i
    model=np.zeros(np.size(x,axis=1)+2)
    for count in range(0,np.size(alphas,axis=0)):
        if alphas[count]>1e-4:
            ay = np.array((alphas[count],y[count]))
            model = np.vstack((model,np.append(ay,x[count])))
    #delete initialization row
    model = np.delete(model,0,axis = 0)
    return model

def gen_model_reg(alphas,x):
#for svm_regressor, returns array (alpha,input_vector)
    N = np.size(x,axis = 0)
    dim = np.size(x,axis=1)
    model=np.zeros((2*N,dim+1))
    for count in range(0,N):
        model[count,0],model[count,1:] = alphas[count],x[count,:]
        model[count+N,0],model[count+N,1:] = alphas[count+N],x[count,:]
    return model

def b_from_model(model,kernel,ker_param):
#For svm classifier
#find b using a support vector. For support_vector, sum_i[a_i*y_i*ker(x_i,x_supp)]+b=y_supp
    sum_ker=0
    for count in range(0,np.size(model,axis=0)):
        sum_ker += model[count,0]*model[count,1]*globals()[kernel](model[count,2:],model[0,2:],ker_param)
    b=model[0,1]-sum_ker
    return b

def svm_eval(x,model,kernel,ker_param):
#hyperplane function. For dual case, f(x) = sum+i[a_i*y_i*kernel(x_i,x)]+b
    b = b_from_model(model,kernel,ker_param)
    temp=0
    for count in range(0,np.size(model,axis=0)):
        temp = temp + model[count,0]*model[count,1]*globals()[kernel](x,model[count,2:],ker_param)
    temp = temp + b
    return temp

def svm_classifier(x,model,kernel,ker_param):
#return +-1 depending on which side of boundary. Trained binary classifier
    return np.sign(svm_eval(x,model,Kernel,ker_param))

def bulk_svm_eval(x_test,model,kernel_pairwise,ker_param,b):
#svm_classifier acting on an array of inputs
    K = globals()[kernel_pairwise](model[:,2:],x_test,ker_param)
    ay = np.multiply(model[:,0],model[:,1])
    f_j = np.sign(np.dot(K.T,ay)+b)
    return(f_j)

def bulk_svm_eval_no_sign(x_test,model,kernel_pairwise,ker_param,b):
#svm_classifier acting on an array of inputs
    K = globals()[kernel_pairwise](model[:,2:],x_test,ker_param)
    ay = np.multiply(model[:,0],model[:,1])
    f_j = np.dot(K.T,ay)+b
    return(f_j)

def bulk_svm_reg_eval(x_test,model,kernel_pairwise,ker_param,b):
#svm_regressor acting on an array of inputs
    N=int(np.size(model,axis=0)/2)
    K = globals()[kernel_pairwise](x_test,model[:N,1:],ker_param)
    alphas = model[:,0]
    diff_a_i = alphas[:N] - alphas[N:]
    f_j = np.dot(K,diff_a_i)+b
    return(f_j)
