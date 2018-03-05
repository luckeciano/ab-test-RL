import tensorflow as tf
import config
import utils
import numpy as np
class Agent():
    def __init__(self):
        reward = np.zeros((config.training_size, 1))
        action = np.zeros((config.training_size, 1))
        self.memory = {"reward": reward, "action": action}
        self.training_set = 0

    def build_agent(self, actions):
        self.actions = actions
        tf.reset_default_graph()

        #These two lines established the feed-forward part of the network. This does the actual choosing.
        self.weights = tf.Variable(tf.ones([actions]))
        self.chosen_action = tf.argmax(self.weights,0)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None, 1],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None, 1],dtype=tf.int32)

        self.responsible_weight = tf.gather(self.weights, self.action_holder)
        self.regularizer = tf.nn.l2_loss(self.responsible_weight)
        self.loss = -(tf.multiply(tf.log(self.responsible_weight), self.reward_holder))
        self.loss = tf.reduce_mean(self.loss + config.lambd*self.regularizer)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        self.update = self.optimizer.minimize(self.loss)

        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)
        return self

    def get_session(self):
        return self.sess

    def get_weights(self):
        return self.sess.run(self.weights)
        
    def get_actions(self):
        return self.actions

    def act(self):
        if utils.choose_action_randomly(config.e):
            return np.random.randint(self.get_actions())
        else:
            return self.sess.run(self.chosen_action)
      
    def train(self,  action_batch, reward_batch):
        return self.sess.run([self.loss,self.update,self.responsible_weight,self.weights], feed_dict={self.reward_holder:reward_batch,
                    self.action_holder:action_batch})

    def aggregate_observation(self, action, reward):
        if self.training_set >= config.training_size:
            return
        self.memory['action'][self.training_set] = action
        self.memory['reward'][self.training_set] = reward
        self.training_set += 1
    
    def get_memory_size(self):
        return self.training_set
    
    def get_memory_actions(self):
        return self.memory['action']
    
    def get_memory_rewards(self):
        return self.memory['reward']
    
    def clear_memory(self):
        self.training_set = 0