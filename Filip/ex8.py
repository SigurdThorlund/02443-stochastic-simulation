# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 14:25:59 2023

@author: filip
"""


import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import scipy.stats as stats
import random
import math



#%% Question 1


def bootstrap(X):
    n = len(X)
    return np.random.choice(X,n),n

r = 100
X = np.array([56, 101, 78, 67, 93, 87, 64, 72, 80, 69])
a,b=-5,5

count=0
for i in range(r):
    sim,n = bootstrap(X)
    mu = np.mean(sim)
    if np.logical_and(a < sum(X/n)-mu,sum(X/n)-mu < b):
        count+=1

P = count/r

print(P)


#%% Question 2


X = [5,4,9,6,21,17,11,20,7,10,21,15,13,16,8]


r = 1000
var = np.zeros(r)
for i in range(r):
    sim,n = bootstrap(X)
    var[i] = np.var(sim,ddof=1)

print(np.var(var,ddof=1))


#%% Question 3



def bootstrapEstimateMed(X,r):
    sample_med = np.median(X)
    n = len(X)
    
    bts = np.random.choice(X,[n,r])
    b_med = np.median(bts,axis=0)
    b_var = np.var(b_med,ddof=1)
    
    return sample_med, b_var
    

N = 200
X = stats.pareto.rvs(1.05,size=N)
r = 100



#%% A
np.mean(X)
np.median(X)


#%% B
def bootstrapEstimateMean(X,r):
    sample_mean = np.mean(X)
    n = len(X)
    
    bts = np.random.choice(X,[n,r])
    b_mean = np.mean(bts,axis=0)
    b_var = np.var(b_mean,ddof=1)
    return sample_mean, b_var

print(bootstrapEstimateMean(X,r))

#%%
print(bootstrapEstimateMed(X, r))

#%%


N = 10000
X = stats.pareto.rvs(1.05,size=N)
r = 100

print("mean: ", bootstrapEstimateMean(X,r))

print("median: ",bootstrapEstimateMed(X,r))


#it is easier to estimate the median than the mean

