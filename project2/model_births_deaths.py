import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


N = 10000
I = 10
S = N-I
R = 0
D = 0
time = 0
B = 0

# CYCLIC BEHAVIOR?
"""beta = 0.04 # governs rate from S to I
gamma = 0.015 # governs rate from I to R
delta = 0.001 # governs rate from R to S
mu = 0.00001 # governs rate from I to D
"""

beta =  0.4
gamma = 0.035
delta = 0.01
mu =    0.0001

natural_death_rate = 0.0001
birth_rate = 0.0001


S_list = [S]
I_list = [I]
R_list = [R]
D_list = [D]
t_list = [time]
B_list = [B]
N_list = [N]


while time < 1000:
    
    # ##Lockdown
    # if np.logical_and(time > 15, time < 100):
    #      beta = 0.001
        
    # elif np.logical_and(time > 100, time < 300):
    #      beta = 0.1
    # else: 
    #     beta = 0.4
    
    N = S+R+I
    rateSI = beta*S*I/N
    rateIR = gamma*I
    rateRS = delta*R
    rateID = mu*I
    rate_bs = birth_rate * N
    rate_nd = natural_death_rate*N

    totRate = rateSI + rateIR + rateRS + rateID + rate_bs + rate_nd
    if(totRate == 0):
        break

    t = stats.expon.rvs(scale = 1/totRate)
    time += t
    
    
    probs = np.array([rateSI, rateIR, rateRS, rateID, rate_bs, rate_nd])/totRate
    event = np.random.choice(6, 1, p = probs)

    if event == 0:
        S -= 1
        I += 1
    elif event == 1:
        I -= 1
        R += 1
    elif event == 2:
        R -= 1
        S += 1
    elif event==3:
        I -= 1
        D += 1
    elif event==4:
        S += 1
        B += 1
    else:
        D+=1
        who = np.random.choice(3,p=np.array([S,I,R])/N)
        if who ==0:
            S-=1
        elif who ==1:
            I-=1
        else:
            R-=1
    
    S_list.append(S)
    I_list.append(I)
    R_list.append(R)
    D_list.append(D)
    t_list.append(time)
    B_list.append(B)
    N_list.append(N)
    print(time)

plt.figure()
plt.plot(t_list, S_list, color = 'blue', label = 'S')
plt.plot(t_list, I_list, color = 'red', label = 'I')
plt.plot(t_list, R_list, color = 'green', label = 'R')
#plt.plot(t_list, D_list, color = 'black', label = 'D')
#plt.plot(t_list, B_list, color = 'purple', label = 'B')
#plt.plot(t_list, np.array(D_list)-np.array(B_list), color = 'orange', label = 'DB diff')

plt.plot(t_list, N_list, color = 'purple', label = 'N')
plt.grid()
plt.legend()
#plt.savefig("BSIRD.png")
plt.show()
