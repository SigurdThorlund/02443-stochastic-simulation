# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:15:13 2023

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




#service time
def service_time(meanST,typ):
    if typ == "exp":
        return stats.expon.rvs(scale=meanST,size=1)
    if typ == "constant":
        return meanST
    if typ == "pareto":
        k=1.05
        return pareto(k,1)
    if typ == "normal":
        nd = stats.norm.rvs(mst,2)
        if nd > 0:  return nd
        else:   return 0
    #if typ == "unif":
    #    return random.random



#arrival
def arrival(meanTBC,typ):
    if typ == "erlang":
        return stats.erlang.rvs(a=1,scale=meanTBC,size=1) #basically exponential
                                        #alternative stats.erlang.rvs(a=4,scale=meanTBC/4,size=1)
    if typ == "pois":
        return stats.expon.rvs(scale=meanTBC,size=1)
    if typ == "hyperexp":
        p1,l1,l2 = 0.8,0.8333,5.0
        return hyperexp(p1, l1, l2)
    
    
arrival(mtbc,"pois")
    
    
def hyperexp(p1, l1, l2):
    U = np.random.random(1)
    if U<=p1:
        return stats.expon.rvs(scale=1/l1,size=1)
    else:
        return stats.expon.rvs(scale=1/l2,size=1)


def pareto(k,N):
    U = np.random.random(N)
    return mst/(k/(k-1)) *(U)**(-1/k)


def sim(N,a_typ,s_typ):
    blocked = 0
    service_unit = np.zeros(m)
    arriveTime = 0
    arrivals = np.zeros(N)

    for i in range(N):
        arrivals[i] = arrival(mtbc,a_typ)
        arriveTime += arrivals[i]
        serviceTime = service_time(mst,s_typ)
        if np.min(service_unit) <= arriveTime:
            for s in range(m):    
                if service_unit[s] <= arriveTime:
                    service_unit[s] = arriveTime + serviceTime
                    break
        else:
            blocked +=1
    pBlock = blocked/N
    return pBlock


print(sim(n,"pois","exp"))

#Analytical solution of pBlock:
A = mtbc*mst
B = (A**m / np.math.factorial(m)) / sum([A**i/np.math.factorial(i) for i in range(m+1)])


def calculateCI(blocks):
    N = len(blocks)
    ci= np.zeros(2)

    #confidence lvel
    CL = 0.975
    #deg of freedom
    DF = N-1

    #this z-value might be wrong
    z = np.abs(t.ppf( (1-CL)/2,DF ) )

    m = np.mean(blocks) 
    s = np.std(blocks,ddof=1)
        
    pm = z * s/np.sqrt(N)
    
    ci = [m-pm, m+pm]
    return(ci)


sims = 10
Blocks = np.zeros(sims)
for i in range(sims):
    Blocks[i] = sim(n,"pois","exp")


expB_CI = (calculateCI(Blocks))

np.mean(Blocks)


plt.scatter(range(10),Blocks)
plt.axhline(expB_CI[0],color='red')
plt.axhline(expB_CI[1],color='green')


#%% Question 2



sim(n,"pois","exp")
sim(n,"erlang","exp")
sim(n,"hyperexp","exp")




#%% Question 3

#arrival, service

sim(n,"hyperexp","exp")




#%% q4


#arrival distributions confidence intervals

sims = 10
B_erl = np.zeros(sims)
for i in range(sims):
    B_erl[i] = sim(n,"erlang","exp")

sims = 10
B_hyp = np.zeros(sims)
for i in range(sims):
    B_hyp[i] = sim(n,"hyperexp","exp")



print(calculateCI(Blocks))
print(calculateCI(B_erl))
print(calculateCI(B_hyp))

plt.scatter([1,1],calculateCI(Blocks))
plt.scatter([2,2],calculateCI(B_erl))
plt.scatter([3,3],calculateCI(B_hyp))
plt.axhline(B)

#service time distributions confidence intervals
sims = 10
B_const = np.zeros(sims)
for i in range(sims):
    B_const[i] = sim(n,"pois","constant")

sims = 10
B_par = np.zeros(sims)
for i in range(sims):
    B_par[i] = sim(n,"pois","pareto")



print(calculateCI(Blocks))
print(calculateCI(B_const))
print(calculateCI(B_par))










