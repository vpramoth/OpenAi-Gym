import random
import gym
import tensorflow as tf
import numpy as np
band = [0.2,0,-0.2,-5]
def bandit(action):
    k=random.uniform(-5, 5)
    if (band[action]<k):
        return 1
    else:
        return -1
rAll = [0 for i in range(4)]
w = tf.Variable(tf.ones(len(band)))
action_taken = tf.placeholder(shape=[1],dtype=tf.int32)
reward = tf.placeholder(shape=[1],dtype=tf.float32)
k = tf.slice(w,action_taken,[1])
loss = -(tf.log(k)*reward)
t = tf.train.GradientDescentOptimizer(learning_rate=0.01)
uc = t.minimize(loss)
init = tf.global_variables_initializer()
e=0.1

with tf.Session() as ss:
    ss.run(init)
    i =0
    while(i<1000):
        if np.random.rand(1) < e:
            action = np.random.randint(len(band))
        else:
            a = tf.argmax(w,0)
            action=ss.run(a)
        r = bandit(action)
        w1 = ss.run([uc],feed_dict={reward:[r],action_taken:[action]})
        rAll[action] = rAll[action]+r
        if i % 50 == 0:
            print(ss.run(w))
            print("Running reward for the " + str(len(band)) + " bandits: " + str(rAll))
        i += 1
    ww = ss.run(w)
k = np.argmax(ww)
print(k)
print("The agent thinks bandit " + str(k + 1) + " is the most promising....")
if np.argmax(ww) == np.argmax(-np.array(band)):
    print("...and it was right!")
else:
    print("...and it was wrong!")

