import gym
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math as m
gamma = 0.99
no_episodes = 1


def discounted_reward(r):  #fn for calculating discounted rewards from rewards

    k =0
    new =[0 for i in range(len(r))]

    for t in range(len(r)-1,-1,-1):
        new[t] = r[t] + gamma*k
        k = new[t]
    return new
# def discounted_reward(r):
#     """ take 1D float array of rewards and compute discounted reward """
#     discounted_r = np.zeros_like(r)
#     running_add = 0
#     for t in reversed(range(0, r.size)):
#         running_add = running_add * gamma + r[t]
#         discounted_r[t] = running_add
#     return discounted_r


class agent():    #defining neural net for predicting the output
    def __init__(self):
        self.state = tf.placeholder(shape=[None,4],dtype=tf.float32)
        self.hidden = slim.fully_connected(self.state,8,activation_fn=tf.nn.relu,biases_initializer=None)    #defining a network of 4*8*2
        self.output = slim.fully_connected(self.hidden,2,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.desired_action = tf.argmax(self.output,1)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        self.reward_holder=tf.placeholder(shape=[None],dtype=tf.float32)
        self.index = tf.range(0, tf.shape(self.output)[0])*(tf.shape(self.output)[1])+self.action_holder
        self.responsible_weights = tf.gather(tf.reshape(self.output, [-1]), self.index)
        self.loss = -tf.reduce_mean(tf.log(self.responsible_weights)*self.reward_holder)
        self.trainer = tf.train.AdamOptimizer(learning_rate=1e-2)
        self.uc = self.trainer.minimize(self.loss)
tf.reset_default_graph()
Myagent = agent()
init = tf.global_variables_initializer()

env = gym.make('CartPole-v0')
# env.render()
rendering = False
with tf.Session() as sess:
    i=0
    sess.run(init)
    e = 1
    total_reward = []
    while i<no_episodes:

        s = env.reset()

        total = []
        j=0
        running_reward = 0
        while j<1000:
            if (i == 100 or i==999):
                env.render()

            # if running_reward >=199 :
            #     env.render()
            #     rendering = True
            # env.render()
            j=j+1
            # a,b = sess.run([Myagent.desired_action,Myagent.output], feed_dict={Myagent.state: [s]})

            a_dist = sess.run(Myagent.output, feed_dict={Myagent.state: [s]})
            a = np.random.choice(a_dist[0], p=a_dist[0])
            a = np.argmax(a_dist == a)
            # if np.random.rand(1) < e:
            #     a[0] = np.random.randint(2)
            s1,r,d,_ = env.step(a)
            total.append([s,a,r])
            s = s1
            running_reward += r


            if d==True:
                total = np.array(total)
                total[:, 2] = discounted_reward(total[:, 2])
                feed_dict = {Myagent.reward_holder: total[:, 2],
                             Myagent.action_holder: total[:, 1], Myagent.state: np.vstack(total[:, 0])}
                _ = sess.run(Myagent.uc, feed_dict=feed_dict)
                total_reward.append(running_reward)
                # e = 1*m.exp(-(i+1)*0.1)
                # e = 1. / ((i / 500) + 1)
                # if i>4000:
                #     e = 0
                break

        if i % 100 == 0:
            print(np.mean(total_reward[-100:]))

        i=i+1
        # a = sess.run(Myagent.output,feed_dict={Myagent.state_in: [s]})





