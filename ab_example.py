from environment import Workspace, Environment
import matplotlib.pyplot as plt
from ab_model import AB_Model
import numpy as np
import math

import tensorflow as tf
from tensorflow.python import debug as tf_debug


nb_timesteps = 100  
nb_batches = 1
batch_size = 1000
training_size = 10000

def distribution_a(timestep):
    return 1.0 + math.sin(2*math.pi*timestep/50.0)

def distribution_b(timestep):
    return 1.0 + math.sin(2*math.pi*timestep/50.0 + math.pi)

def conversion_a(timestep):
    prob = distribution_a(timestep)/100.0
    return np.random.choice([0,1], size = (1), p = [1 - prob, prob])

def conversion_b(timestep):
    prob = distribution_b(timestep)/100.0
    return np.random.choice([0,1], size = (1), p = [1 - prob, prob])

def shuffle_batch(train, test):
    permutation=np.random.permutation(training_size)
    permutation=permutation[0:batch_size]
    return train[permutation], test[permutation]

def main():
    #-----------------------------------HYPERPARAMETERS-----------------------------------------
    e = 0.1
    lambd = 1.0
    #------------------------------------ENVIRONMENT---------------------------------------------
    a = Workspace(conversion_a)
    b = Workspace(conversion_b)

    workspaces = []
    workspaces.append(a)
    workspaces.append(b)

    env = Environment(workspaces)
    
    dist_a = {'X': [], 'Y':[]}
    dist_b = {'X':[], 'Y':[]}


    #--------------------------------------------AGENT--------------------------------------------
    tf.reset_default_graph()

    #These two lines established the feed-forward part of the network. This does the actual choosing.
    weights = tf.Variable(tf.ones([len(workspaces)]))
    chosen_action = tf.argmax(weights,0)

    #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
    #to compute the loss, and use it to update the network.
    reward_holder = tf.placeholder(shape=[None, 1],dtype=tf.float32)
    action_holder = tf.placeholder(shape=[None, 1],dtype=tf.int32)

    responsible_weight = tf.gather(weights, action_holder)
    regularizer = tf.nn.l2_loss(responsible_weight)
    loss = -(tf.multiply(tf.log(responsible_weight), reward_holder))
    loss = tf.reduce_mean(loss + lambd*regularizer)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    update = optimizer.minimize(loss)


    #----------------------------------------------------------------------------------------------

    
    log = {'X': [], 'Y':[]}

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        for i in range(nb_timesteps):
            print("TIMESTEP " + str(i))
            dist_a['Y'].append(distribution_a(i))

            dist_b['Y'].append(distribution_b(i))
            actions_tensor = np.zeros((training_size, 1))
            rewards_tensor = np.zeros((training_size,1))
            for j in range(training_size):
                action_elem = np.zeros(1)
                reward_elem = np.zeros(1)
                if np.random.rand(1) < e:
                    action_elem = np.random.randint(len(workspaces))
                else:
                    action_elem = sess.run(chosen_action)
                reward_elem = env.act(action_elem, i)
                actions_tensor[j][0] = action_elem
                rewards_tensor[j][0] = reward_elem
            for j in range(nb_batches):
                action_batch, reward_batch = shuffle_batch(actions_tensor, rewards_tensor)
                a,_,resp,ww = sess.run([loss,update,responsible_weight,weights], feed_dict={reward_holder:reward_batch,
                    action_holder:action_batch})
            print(a)
            print(ww)

            total_reward = np.sum(rewards_tensor)
            reward_mean = float(total_reward)/float(training_size)
            print("Total Reward of timestep " + str(i) + ': ' + str(reward_mean))
            log['X'].append(i)
            log['Y'].append(100.0*reward_mean)        

        plot_timestep = np.arange(nb_timesteps)
        
        fig, ax = plt.subplots()
        plt.plot(log['X'], log['Y'], 'o')
        ax.plot(plot_timestep, dist_a['Y'], label = 'Workspace A')
        ax.plot(plot_timestep, dist_b['Y'], label = 'Workspace B')
        legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
        plt.show()


main()


