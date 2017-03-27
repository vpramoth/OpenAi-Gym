import gym
import random
from random import randrange
import matplotlib.pyplot as plt
import numpy as np
import math as m
env = gym.make('FrozenLake-v0')
# for _ in range(100):
#     env.render()
no_episodes = 5000
q = np.zeros([env.observation_space.n,env.action_space.n])
tr = 0
# print(env.action_space.n)
# for i in range(no_episodes):
i = 0
rList = []
while(i<=700):
    n = []
    s = env.reset()
    tr = 0
    d=False
    j=0
    while j<50:

        a = np.argmax(q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        # k = random.uniform(0, 1)
        # if(k<(1-(0.5*m.exp(-i/1000)))):
        #     a = np.argmax(q[s,:])
        # else:
        #     a = random.randint(0,len(q[s,:])-1)
        # print(a)
        s1,r,d, _ = env.step(a)
        print("os,a,r,ns")
        print(s,a,r,s1)

        if (s in ([c[0] for c in n]) and a in ([c[1] for c in n])):
            pass
        else:
            n.append([s, a, tr, 1])
        tr = tr + r
        j = j + 1
        if d==True:
            break
        s = s1
        # print("ss")
        # print(s)
    print(tr)#gamma =1, Iam not reducing the reward
    for w in range(len(n)):
        q[n[w][0],n[w][1]] = q[n[w][0],n[w][1]]+ 0.05*(((tr-n[w][2])-q[n[w][0],n[w][1]]))

    i=i+1
    rList.append(tr)
    sco = sum(rList) / (i + 1)
    plt.scatter(i, sco)

# print('no. of episodes')
plt.show()
print("Score over time: " +  str(sum(rList)/i))
print(n)
print(q)