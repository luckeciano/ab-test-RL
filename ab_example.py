from environment import Workspace, Environment
import numpy as np
import math

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import config
from logger import Logger
from agent import Agent
import utils

def distribution_a(timestep):
    return 1.0 + math.sin(2*math.pi*timestep/100.0)

def distribution_b(timestep):
    return 1.0 + math.sin(2*math.pi*timestep/100.0 + math.pi)

def conversion_a(timestep):
    prob = distribution_a(timestep)/100.0
    return np.random.choice([0,1], size = (1), p = [1 - prob, prob])

def conversion_b(timestep):
    prob = distribution_b(timestep)/100.0
    return np.random.choice([0,1], size = (1), p = [1 - prob, prob])


def main():


    logger = Logger()
    #------------------------------------ENVIRONMENT---------------------------------------------
    a = Workspace(conversion_a)
    b = Workspace(conversion_b)

    workspaces = []
    workspaces.append(a)
    workspaces.append(b)

    env = Environment(workspaces)
    
    #-------------------------------------------------------------------------------------------
    agent = Agent().build_agent(len(workspaces))
    sess = agent.get_session()

    logger.create_dataholder("Target")
    logger.create_dataholder("Workspace_A")
    logger.create_dataholder("Workspace_B")

  
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    for i in range(config.nb_timesteps):
        Logger.write("INFO", "TIMESTEP " + str(i))
        logger.add_datapoint("Workspace_A", i, distribution_a(i))
        logger.add_datapoint("Workspace_B", i, distribution_b(i))

        actions_tensor = np.zeros((config.training_size, 1))
        rewards_tensor = np.zeros((config.training_size,1))
        
        for j in range(config.training_size):
            action_elem = np.zeros(1)
            reward_elem = np.zeros(1)
            action_elem = agent.act()
            reward_elem = env.act(action_elem, i)
            actions_tensor[j][0] = action_elem
            rewards_tensor[j][0] = reward_elem
        
        for j in range(config.nb_batches):
            action_batch, reward_batch = utils.shuffle_batch(actions_tensor, rewards_tensor)
            loss_value,upd,resp,ww = agent.train(action_batch, reward_batch)
        
        Logger.write("INFO", str(loss_value))
        Logger.write("INFO", str(ww))

        total_reward = np.sum(rewards_tensor)
        reward_mean = float(total_reward)/float(config.training_size)
        
        Logger.write("INFO", "Total Reward of timestep " + str(i) + ': ' + str(reward_mean))
        
        logger.add_datapoint("Target", i, 100.0*reward_mean)             
    
    logger.init_plot()
    logger.plot("Target", 'o')
    logger.plot("Workspace_A", linestyle = None)
    logger.plot("Workspace_B", linestyle = None)
    logger.show()


main()


