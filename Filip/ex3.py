import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import scipy.stats as stats
import random
import math

from scipy.stats import expon
import matplotlib.pyplot as plt

from scipy.stats import t


n = 10
#%% Exponential distribution 

lmd = 1.3

#Simulation
exp_sim = -np.log(np.random.random(n))/lmd

#Analytical
exp_a = stats.expon.rvs(scale=lmd, size=n)

#plots
plt.figure()
plt.hist(exp_sim,   color="blue",               density=True,rwidth=2)
plt.hist(exp_a,     color="red",    alpha=0.7,  density=True,rwidth=2)




#%% Normal dist

u1 = np.random.rand(n)
u2 = np.random.rand(n)

x1 = np.sqrt(-2*np.log(u1) ) * np.cos(2*np.pi*u2)
x2 = np.sqrt(-2*np.log(u1) ) * np.sin(2*np.pi*u2)

xx = np.concatenate([x1,x2])



plt.figure()
plt.hist(xx,density=True,color="blue")
nd = stats.norm.rvs(size=n)
plt.hist(nd, color='black',density=True,alpha=0.5)


#%% 3





ci = np.zeros((100,4))

#confidence lvel
CL = 0.999
#deg of freedom
DF = n-1


#this z-value might be wrong
z = np.abs(t.ppf( (1-CL)/2,DF ) )

for i in range(100):
    u1 = np.random.rand(n)
    u2 = np.random.rand(n)
    x1 = np.sqrt(-2*np.log(u1) ) * np.cos(2*np.pi*u2)
    #x1 = np.random.normal(0,1,n)
    m = np.mean(x1) 
    s = np.std(x1)
    
    pm = z * s/np.sqrt(n)

    ci[i,:] = [m+pm, m-pm,m,s]


plt.plot(ci[1:100,:2])
plt.plot(ci[1:100,2],color="gray")

plt.ylim(-2,2)





