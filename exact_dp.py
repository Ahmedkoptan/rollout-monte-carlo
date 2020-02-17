#variable travelling cost (in terms of time index)
#random parking cost


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!



#CONSTANTS
myrange = 200
b = 100
a = 10
TRAVEL = a/2              #travelling cost
SEED = 40
ACTIONS = ['park','move right']
np.random.seed(SEED)

class Driver:
    def __init__(self, pf, f, c, N, gamma):
        self.pf = pf
        self.f = f
        self.c = c
        self.N = N
        self.gamma = gamma

    def calc_costs(self):
        # initialize cost array
        Jt = np.zeros(self.N + 1, float)
        Jt[self.N] = c[self.N]

        for i in range(self.N-1,-1,-1):
            if i == self.N-1:
                Jt[i] = (self.pf[i]*min(self.c[i],self.c[i+1])) + ((1-self.pf[i])*self.c[i+1])
            else:
                Jt[i] = (self.pf[i]*min(self.c[i],Jt[i+1])) + ((1-self.pf[i])*Jt[i+1])
        #print('Current optimal costs matrix is:\n',Jt)
        return Jt

    def prob_estimate(self,k, status):
        r = status[status != 2]
        R = np.mean(r)
        for i in range(k+1,N,1):
            self.pf[i] = (self.gamma*pf[i]) + ((1-self.gamma)*R)


    def park(self):
        i = 0           #time/parking spot index
        park = 0        #park
        mr = 1          #move right
        action = mr     #move right
        parked=False    #while loop boolean

        #define status array
        status = np.zeros_like(self.f)+2

        while not parked and i<=self.N:

            # at garage
            if i==self.N:
                # park
                action = park

            # any other spot
            else:
                # if free
                if self.f[i] == 1:
                    status[i] = 1
                    self.pf[i] = 1.0
                    self.prob_estimate(i,status)

                    Jt = self.calc_costs()
                    # compare cost with optimal cost of right and left
                    if self.c[i] <= Jt[i + 1]:
                        # park if better than or equal to both
                        action = park
                    # else cost of current spot cost is not better than optimal of right
                    else:
                        # move right
                        action = mr

                # else current spot is not free
                else:
                    self.pf[i] = 0.0
                    status[i] = 0
                    self.prob_estimate(i,status)

                    # move right
                    action = mr

            print('iteration',i,'action chosen is',ACTIONS[action])


            # if action is to park
            if action == park:
                print('Parked at location',i+1,'with cost',self.c[i])
                # set boolean to stop while loop
                parked = True
                cost = self.c[i]

            # else move right
            else:
                # increment time index
                i += 1
        return cost

#number of parking spaces (without garage)
myN = np.linspace(1,myrange,myrange,dtype=int)

costs = np.zeros_like(myN,float)

#PF = pf = np.random.rand(myrange)
#F = np.random.randint(0, 2, myrange)
#F = np.zeros_like(PF,int)
#F[np.where(PF>0.5)]=1

gamma = 0.7

#for i in range(len(myN)):

N = 200

c = np.zeros(N + 1)
for co in range(len(c)):
    c[co] = (co + 1 - int(myrange * 0.5)) ** 2
print('Cost of every spot is:\n', c)
# space actually free or not
#f = np.random.randint(0, 2, N)
#f = F[:i+1].copy()
# probability of space being free
pf = np.random.rand(N)
#pf = PF[:i+1].copy()
print('Probability of every spot being free is:\n', pf)
# space actually free or not
f = np.zeros_like(pf,int)
f[np.where(pf>0.3)]=1
f[np.argmin(c)] = 1
data = pd.concat([pd.DataFrame(c[:-1]),pd.DataFrame(pf),pd.DataFrame(f)],axis=1)
print('Whether every spot is actually free or not:\n', f)
driver = Driver(pf,f,c,N,gamma)
costs = driver.park()


print('\n\n')


# Plot the surface.
fig = plt.figure()
plt.plot(np.array([*myN,201]),c)

plt.show()
data


