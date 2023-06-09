
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc



def LCG(M,a,c,x0,L):
   numbers = np.zeros(L)
   
   for i in range(L):
       numbers[i] = (a*x0+c) %M
       x0 = numbers[i]
   return numbers



M,a,c,x0,L = 16,5,1,3,10000

randomList = LCG(M,a,c,x0,L)
randoms = randomList/M



# plt.figure()
# plt.scatter(range(L),randomList)
# #plt.ylim(0,20)



# plt.figure()
# plt.hist(randoms,bins=10,rwidth=0.95)

# plt.figure()
# plt.scatter(range(L),randoms)


###CHI SQUARE TEST
def chisq_test(randoms):
    
    n_classes = 10
    n_observed = (np.histogram(randoms,n_classes)[0])
    
    n_expected = L/n_classes
    
    if n_expected < 5:
        print("Warning: low number of n_expected")
    
    #test statistic
    T = sum(((n_observed - n_expected)**2)/n_expected)
    
    
    p = 1-sc.chi2.cdf(T , df=9)
    
    #test
    #print(sc.chisquare(n_observed)[1])
    
    print("Chisq test statistic: ",T,". p-value: ",p)
    if p<0.05:
        print("Hypothesis is rejected as p<0.05 (no significant difference)")
    else:
        print("A significant difference is detected as p>0.05)")
    return (T,p)

chisq_test(randoms)

###KOLMOGOROV SMIRNOV TEST
def ks_test(randoms):
    randomsSorted = np.sort(randoms)
    expected =np.linspace(0, 1, L)
    
    plt.figure()
    plt.step(randomsSorted,np.linspace(0, 1, L),where='post')
    plt.plot(expected,np.linspace(0, 1, L))
     
    T_ks = max(abs(randomsSorted-expected))
    #p_ks = 
    return T_ks #,p_ks
    
    #test
    #return sc.kstest(randomsSorted, "uniform")
    #return sc.kstest(randomsSorted, expected)


ks_test(randoms)



# mdn = np.median(randoms)
# n1 = len(randoms[randoms > mdn])
# n2 = len(randoms[randoms < mdn])










def run_test1(randoms):
    mdn = np.median(randoms)
    n1 = len(randoms[randoms > mdn])
    n2 = len(randoms[randoms < mdn])
    Ra = 0
    Rb = 0
    for i in range(1,L):
        if randoms[i] > mdn and randoms[i-1] < mdn:
            Ra += 1
        elif randoms[i] < mdn and randoms[i-1] > mdn:
            Rb += 1
    if randoms[0] > mdn: Ra +=1 
    else: Rb+=1
    
    T_r1 = Ra + Rb

    mn = (2*(n1*n2)/(n1+n2) + 1)
    sd = (2*(n1*n2*(2*n1*n2-n1-n2))/((n1+n2)**2*n1+n2-1) )

    p_r1 = 1 - sc.norm.cdf(T_r1,mn,sd)
    return T_r1,p_r1
    
run_test1(randoms)


