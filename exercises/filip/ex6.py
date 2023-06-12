# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 11:09:58 2023

@author: filip
"""

import numpy as np
import scipy.stats as stats
from scipy.stats import t
import matplotlib.pyplot as plt


n=10000
m = 10      #service units
mst = 8     #mean service time
mtbc = 1    #mean time between customers


def poisson():
    P = [mst**i/ np.math.factorial(i) for i in range(m+1)]
    return P/np.sum(P)



def g(x):
    A=mst
    return A**x /np.math.factorial(x)
    

def Metropolis_Hasting(N):
    X = np.zeros(N) #states
    for i in range(N-1):
        Y = np.random.randint(0, m+1) #dx sampled from some symmetric dist
        
        if g(Y) >= g(X[i]):
            X[i+1] = Y
        
        elif (g(Y) <= g(X[i]) ) and (np.random.random() <= g(Y)/g(X[i]) ):
            X[i+1] = Y
        else:
            X[i+1] = X[i]
    return X

r = Metropolis_Hasting(n)

plt.plot(poisson(),color='red')
plt.hist(r,bins=[-0.5, 0.5, 1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,  9.5, 10.5],density=1,edgecolor="white")




An = poisson()

pvals = np.zeros(100)

sampL = 4
for i in range(100):
    run = Metropolis_Hasting(n)[::sampL]
    
    obs = np.zeros(m+1)
    s = 0
    for k in np.unique(run):
        obs[s] = sum(run == k)
        s+=1
    pvals[i] = stats.chisquare(obs,An*n/sampL)[1]

#plt.hist(run,bins=11,density=1)
#plt.plot(obs,color="red")


plt.figure()
plt.hist(pvals)




#%% Question 2a & 2b

m=10
A1,A2 = 4,4

def g2(i,j):
    if i+j > 10:
        return 0
    return A1**i /np.math.factorial(i) * A2**j /np.math.factorial(j)


def poisson2():
    P = np.zeros((m+1,m+1))
    for i in range(m+1):
        P[i,:] = [g2(i,j) for j in range(m+1)]
    return P/np.sum(P)


def showAnalytical():
    plt.figure()
    pA = poisson2()
    plt.imshow(pA)
    plt.colorbar()
    return
    
def showPlot(rr):
    #MH direct
    plt.figure()
    pHS = np.histogram2d(x=rr[:,0], y=rr[:,1],density=True,bins=11)[0]
    plt.imshow(pHS)
    plt.colorbar()
    return

def Metropolis_Hasting_2Direct(N):
    X = np.zeros((N,2)) #states
    for i in range(N-1):
        Y1 = np.random.randint(0, m+1)
        Y2 = np.random.randint(0, m+1-Y1)
        
        if g2(Y1,Y2) >= g2(X[i,0],X[i,1]):
            X[i+1,:] = Y1,Y2
        
        elif (g2(Y1,Y2) <= g2(X[i,0],X[i,1]) ) and (np.random.random() <= g2(Y1,Y2)/g2(X[i,0],X[i,1]) ):
            X[i+1,:] = Y1,Y2
        else:
            X[i+1,:] = X[i,:]
    return X
r2 = Metropolis_Hasting_2Direct(n)



#Analytical
showAnalytical()
showPlot(r2)


def sample2Values(U):
    List = np.zeros((66,2))
    k=0
    for j in range(m+1):
        for i in range(m+1-j):
            List[k,:] = [i,j]
            k+=1
    return List[np.int(np.random.random()*66),:]


def Metropolis_Hasting_2Coord(N):
    X = np.zeros((N,2)) #states
    Y1,Y2 = sample2Values(np.random.random())
    switch = True
    for i in range(N-1):
        if g2(Y1,Y2) >= g2(X[i,0],X[i,1]):
            X[i+1,:] = Y1,Y2
        
        elif (g2(Y1,Y2) <= g2(X[i,0],X[i,1]) ) and (np.random.random() <= g2(Y1,Y2)/g2(X[i,0],X[i,1]) ):
            X[i+1,:] = Y1,Y2
        else:
            X[i+1,:] = X[i,:]
            
        if switch == True:
            Y1 =np.random.randint(0, m+1-Y2)
        else:
            Y2 =np.random.randint(0, m+1-Y1)
        switch = (switch == False)
    return X

r3 = Metropolis_Hasting_2Coord(n)
showPlot(r3)






#pvalues tests
sims = 100
n = 10000

UniqList = np.zeros((66,2))
k=0
for j in range(m+1):
    for i in range(m+1-j):
        UniqList[k,:] = [i,j]
        k+=1

An2 = poisson2()
An2 = An2[An2!=0]
pvals2 = np.zeros(sims)

sampL2 = 4
for i in range(sims):
    run = Metropolis_Hasting_2Direct(n)[::sampL2]
    
    obs = np.zeros(66)
    for k in range(66):#np.unique(run,axis=0):
        for rr in range(np.shape(run)[0]):
            obs[k] += np.logical_and(UniqList[k,0] == run[rr,0], UniqList[k,1] == run[rr,1])

    pvals2[i] = stats.chisquare(obs,An2*n/sampL2)[1]

plt.figure()
plt.plot(obs)
plt.plot(An2*n/sampL2)

plt.figure()
plt.hist(pvals2)




#%% Question 









