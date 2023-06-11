"""
@author: filip
"""

import numpy as np
import scipy.stats as stats
from scipy.stats import t
import matplotlib.pyplot as plt



n=100




def CI(samples):
    N = len(samples)
    CI = np.zeros(3)
    CL = 0.95
    DF = N-1
    z = np.abs(t.ppf( (1-CL)/2,DF ) )
    m = np.mean(samples)
    s = np.std(samples,ddof=1)
    pm = z * s/np.sqrt(N)
    
    CI = [m-pm,m, m+pm]
    return CI


def crude_estimator(N):
    I = 100
    X=np.zeros(N)
    for i in range(N):
        x = np.random.random(100) #100 = estimation precision
        X[i] = sum(np.exp(x))/N
    return  X


t1 = CI(crude_estimator(n))






def antithetic_var(N):
    Y = np.zeros(N)
    for i in range(N):
        U = np.random.random(N)
        ev = np.exp(U)
        Y[i] = sum((ev + np.exp(1)/ev)/2)/N
    return Y
    
t2 = CI(antithetic_var(n))

def control_var(N):
    U = np.random.random(N)
    X = np.exp(U)
    np.mean(U)
    
    co = np.mean(U*np.exp(U)) - np.mean(U)*np.mean(np.exp(U))
    #va = sum(U - np.mean(U) )**2 / (N-1)
    va = np.var(U)
    c = -co/va
    Z = X + c*(U-(1/2))
    return Z

t3 = CI(control_var(n))



def stratified(N):
    I = 10
    I,N = 10,n
    U = np.zeros((I,N))
    for i in range(I):
        U[i,:] = np.random.random(N)
    
    W = np.zeros(I)
    W = sum(np.exp(U[:,nn]/N + nn/N) for nn in range(N))/N
    return W


t4 = CI(stratified(n))





import timeit

s1 = timeit.default_timer()
t1 = crude_estimator(n)
e1 = timeit.default_timer()

s2 = timeit.default_timer()
t2 = antithetic_var(n)
e2 = timeit.default_timer()

s3 = timeit.default_timer()
t3 = control_var(n)
e3 = timeit.default_timer()

s4 = timeit.default_timer()
t4 = stratified(n)
e4 = timeit.default_timer()

print(e1-s1)
print(e2-s2)
print(e3-s3)
print(e4-s4)







#%%exercise 5

n=100

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
    # if typ == "hyperexpNew":
    #     p1,l1,l2 = 0.8,0.8333,5.0
    #     return hyperexpNew(p1, l1, l2)
    
def hyperexp(p1, l1, l2):
    U = np.random.random(1)
    if U<=p1:
        return stats.expon.rvs(scale=1/l1,size=1)
    else:
        return stats.expon.rvs(scale=1/l2,size=1)


# def hyperexpNew(p1, l1, l2):
#     U = np.random.random(1)
#     return stats.expon.ppf(U,scale=1/l1)*0.8+stats.expon.ppf(U,scale=1/l1)*0.2



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
    return pBlock, np.mean(arrivals)


def control_var5(N):    
    X_a = np.zeros(N)
    X_b = np.zeros(N)
    for i in range(N):
        X_b[i],X_a[i] = sim(10000,"pois","exp")
    co = np.cov(X_a,X_b)[0,1]
    
    vaXA = np.var(X_a)
    vaXB = np.var(X_b)
    c = -co/vaXA
    Z = X_b + c*(X_a-mtbc)
    vaZ = np.var(Z)
    return CI(Z),CI(X_b),vaZ,vaXB


e5Res = control_var5(10)



#%% Question 6
# Some of the functions are updated to include predefined random numbers
#NOTE that only "exp" service time is supported (also constant)



np.random.seed(seed=100)

#service time
def service_time(meanST,typ,rand):
    if typ == "exp":
        return stats.expon.ppf(scale=meanST,q=rand)
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
def arrival(meanTBC,typ,rand):
    if typ == "pois":
        return stats.expon.ppf(scale=meanTBC,q=rand)
    if typ == "hyperexp":
        p1,l1,l2 = 0.8,0.8333,5.0
        return hyperexp(p1, l1, l2,rand)
    # if typ == "hyperexpNew":
    #     p1,l1,l2 = 0.8,0.8333,5.0
    #     return hyperexpNew(p1, l1, l2)
    
def hyperexp(p1, l1, l2,rand):
    U = np.random.random(1)
    if U<=p1:
        return stats.expon.ppf(scale=1/l1,q=rand)
    else:
        return stats.expon.ppf(scale=1/l2,q=rand)

def sim(N,a_typ,s_typ):
    np.random.seed(100)
    
    randoms1 = np.random.random(N)
    randoms2 = np.random.random(N)
    
    blocked = 0
    service_unit = np.zeros(m)
    arriveTime = 0
    arrivals = np.zeros(N)

    for i in range(N):
        arrivals[i] = arrival(mtbc,a_typ,randoms1[i])
        arriveTime += arrivals[i]
        serviceTime = service_time(mst,s_typ,randoms2[i])
        if np.min(service_unit) <= arriveTime:
            for s in range(m):    
                if service_unit[s] <= arriveTime:
                    service_unit[s] = arriveTime + serviceTime
                    break
        else:
            blocked +=1
    pBlock = blocked/N
    return pBlock, np.mean(arrivals)


theta1 = sim(10000,"pois","exp")[0]
theta2 = sim(10000,"hyperexp","exp")[0]

print(theta2 - theta1)



#%% Question 7




def crude_estimator7(N,a):
    I = 100
    P = np.ones(N)
    for i in range(N):
        x = stats.norm.rvs(size=100) #100 = estimation precision
        P[i] = sum(x>a)/N
    return P

CI(crude_estimator7(100,2))


def important_samp(N,a,sig):
    Y = stats.norm.rvs(a,sig,size=N)
    h = Y > a
    
    g = stats.norm.pdf(Y,a,sig)
    f = stats.norm.pdf(Y)
    
    Z = h*f/g
    return Z

CI(important_samp(10000,2,1))

1-stats.norm.cdf(a)



#%% Question 9

def important_samp9(N,k):
    Y = stats.pareto.rvs(k-1,size=N)
    h = Y
    
    g = stats.pareto.pdf(Y,k-1)
    f = stats.pareto.pdf(Y,k)
    
    Z = h*f/g
    return Z

CI(important_samp9(100,1.05))
