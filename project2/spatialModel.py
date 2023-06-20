

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


from scipy.spatial.distance import euclidean



def random_coordinates(nn):
    return np.random.randint(0,100+1,[nn,2])


def distanceMatrix(cdn):
    nn = cdn.shape[0]
    dm = np.zeros((nn,nn))
    
    for i in range(nn):
        dm[i,:] = [euclidean(cdn[i],cdn[j]) for j in range(nn)]
    return dm



N = 200
I = 2
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


rc = random_coordinates(N)
stateList = np.zeros(N)
mean_dist = 5
dist = distanceMatrix(rc)

Is = np.zeros(N).astype(bool)
Is[:I] = True

Ss = np.zeros(N).astype(bool)
Ss[I:] = True

Rs = np.zeros(N).astype(bool)
Ds = np.zeros(N).astype(bool)


beta =  0.934
gamma = 0.015
delta = 0.001
mu =    0.0001



S_list = [Ss.copy()]
I_list = [Is.copy()]
R_list = [Rs.copy()]
D_list = [Ds.copy()]
t_list = [time]
#B_list = [B]
#N_list = [N]


rateSI = np.zeros(N)
rateIR = np.zeros(N)
rateRS = np.zeros(N)
rateID = np.zeros(N)

while time < 1000:
    
    for i in range(N):
        idx = dist[i] < 10
        
        
        rateSI[i] = beta*sum(Ss[idx])*sum(Is[idx]) / sum(idx)
        rateIR[i] = gamma*sum(Is[idx])
        rateIR[Is]
        
        rateID[i] = mu*sum(Is[idx])
        rateRS[i] = delta*sum(Rs[idx])
    rateID = Is * rateID
    rateIR = Is * rateIR
    rateSI = Ss * rateSI
    rateRS = Rs * rateRS
    
    #rateSI = beta*S*I/N
    #rateIR = gamma*I
    
    

    totRate = np.sum(rateSI + rateIR + rateRS + rateID )
    if(totRate == 0):
        break

    t = stats.expon.rvs(scale = 1/totRate)
    time += t
    probs = np.array([rateSI, rateIR, rateRS, rateID],dtype='float64').flatten()/totRate
    event = np.random.choice(4*N, 1, p = probs)

    if event < N:
        Ss[event%N] = False
        Is[event%N] = True
    elif event < 2*N:
        Is[event%N] = False
        Rs[event%N] = True
    elif event < 3*N:
        Rs[event%N] = False
        Ss[event%N] = True
    else:
        Is[event%N] = False
        Ds[event%N] = True
    
    S_list.append(Ss.copy())
    I_list.append(Is.copy())
    R_list.append(Rs.copy())
    D_list.append(Ds.copy())
    t_list.append(time)
    #N_list.append(N)
    #print(time,S,sum(Is),R,D)
    print(time)

plt.figure()
plt.plot(t_list, np.sum(S_list,axis=1), color = 'blue', label = 'S')
plt.plot(t_list, np.sum(I_list,axis=1), color = 'red', label = 'I')
plt.plot(t_list, np.sum(R_list,axis=1), color = 'green', label = 'R')
plt.plot(t_list, np.sum(D_list,axis=1), color = 'black', label = 'D')

plt.grid()
plt.legend()
plt.show()
#plt.plot(t_list, B_list, color = 'purple', label = 'B')
#plt.plot(t_list, np.array(D_list)-np.array(B_list), color = 'orange', label = 'DB diff')




import time

time.sleep(2)
for t in range(0,len(t_list),10):
    plt.figure()
    plt.scatter(rc[:,0][S_list[t]], rc[:,1][S_list[t]],color="blue" )
    plt.scatter(rc[:,0][I_list[t]], rc[:,1][I_list[t]],color="red" )
    plt.scatter(rc[:,0][R_list[t]], rc[:,1][R_list[t]],color="green" )
    plt.scatter(rc[:,0][D_list[t]], rc[:,1][D_list[t]],color="black" )
    plt.title("t="+str(np.round(t_list[t])))
    
    plt.show()
    time.sleep(0.1)
    


