import numpy as np
import config
def shuffle_batch(train, test):
    permutation=np.random.permutation(config.training_size)
    permutation=permutation[0:config.batch_size]
    return train[permutation], test[permutation]

def choose_action_randomly(e):
    return np.random.rand(1) < e