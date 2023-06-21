
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
N = 10000
S = 9990
I = 10
R = 0
D = 0
time = 0

# CYCLIC BEHAVIOR?
"""beta = 0.04 # governs rate from S to I
gamma = 0.015 # governs rate from I to R
delta = 0.001 # governs rate from R to S
mu = 0.00001 # governs rate from I to D
"""

beta = 0.4
gamma = 0.035
delta = 0.01
mu = 0.0001

S_list = [S]
I_list = [I]
R_list = [R]
D_list = [D]
t_list = [time]

while time < 1000:
    ##Lockdown
    if np.logical_and(time > 15, time < 100):
         beta = 0.001
        
    elif np.logical_and(time > 100, time < 300):
         beta = 0.1
    else: 
        beta = 0.4
    
    rateSI = beta*S*I/N
    rateIR = gamma*I
    rateRS = delta*R
    rateID = mu*I

    
    totRate = rateSI + rateIR + rateRS + rateID
    if(totRate == 0):
        break

    t = stats.expon.rvs(scale = 1/totRate)
    time += t
    probs = np.array([rateSI, rateIR, rateRS, rateID])/totRate
    event = np.random.choice(4, 1, p = probs)

    if event == 0:
        S = S - 1
        I = I + 1
    elif event == 1:
        I = I - 1
        R = R + 1
    elif event == 2:
        R = R - 1
        S = S + 1
    else:
        I = I - 1
        D = D + 1

    S_list.append(S)
    I_list.append(I)
    R_list.append(R)
    D_list.append(D)
    t_list.append(time)

plt.figure()
plt.plot(t_list, S_list, color = 'blue', label = 'S')
plt.plot(t_list, I_list, color = 'red', label = 'I')
plt.plot(t_list, R_list, color = 'green', label = 'R')
plt.plot(t_list, D_list, color = 'black', label = 'D')
plt.xlabel("Time [days]")
plt.ylabel("Number of people")

plt.fill_between(np.arange(15, 100, 1),0,10000,alpha=0.8,color="gray",label="lockdown")
plt.fill_between(np.arange(100, 300, 1),0,10000,alpha=0.9,color="lightgray",label="Restrictions")
#plt.axvline(15,color="gray",label="Lockdown",alpha=0.7)
#plt.axvline(100,color="gray",alpha=0.7)
plt.grid()
plt.legend(loc="upper right")
plt.savefig("base.png")
#plt.show()
#S, I, R, D
#(60286, 0, 0, 39714)



