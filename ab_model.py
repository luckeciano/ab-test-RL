# %matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add, Dropout
from keras.optimizers import SGD
from keras import regularizers


import keras.callbacks as cbks
import keras.backend as K

class Gen_Model():
	def __init__(self, learning_rate, input_dim, output_dim):
		self.learning_rate = learning_rate
		self.input_dim = input_dim
		self.output_dim = output_dim

	def predict(self, x):
		return self.model.predict(x)

	def fit(self, states, rewards, epochs, verbose, validation_split, batch_size):
		return self.model.fit(states, rewards, epochs=epochs, verbose=verbose, validation_split = validation_split, batch_size = batch_size)

	def write(self, game, version):
		self.model.save(run_folder + 'models/version' + "{0:0>4}".format(version) + '.h5')

	def read(self, game, run_number, version):
		return load_model( run_archive_folder + game + '/run' + str(run_number).zfill(4) + "/models/version" + "{0:0>4}".format(version) + '.h5', custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})

	def viewLayers(self):
		layers = self.model.layers
		for i, l in enumerate(layers):
			x = l.get_weights()
			print('LAYER ' + str(i))

			try:
				weights = x[0]
				s = weights.shape

				fig = plt.figure(figsize=(s[2], s[3]))  # width, height in inches
				channel = 0
				filter = 0
				for i in range(s[2] * s[3]):

					sub = fig.add_subplot(s[3], s[2], i + 1)
					sub.imshow(weights[:,:,channel,filter], cmap='coolwarm', clim=(-1, 1),aspect="auto")
					channel = (channel + 1) % s[2]
					filter = (filter + 1) % s[3]

			except:
	
				try:
					fig = plt.figure(figsize=(3, len(x)))  # width, height in inches
					for i in xrange(len(x)):
						sub = fig.add_subplot(len(x), 1, i + 1)
						if i == 0:
							clim = (0,2)
						else:
							clim = (0, 2)
						sub.imshow([x[i]], cmap='coolwarm', clim=clim,aspect="auto")
						
					plt.show()

				except:
					try:
						fig = plt.figure(figsize=(3, 3))  # width, height in inches
						sub = fig.add_subplot(1, 1, 1)
						sub.imshow(x[0], cmap='coolwarm', clim=(-1, 1),aspect="auto")
						
						plt.show()

					except:
						pass

			plt.show()

import tensorflow as tf
def custom_loss(y_true, y_pred):
		return tf.metrics.mean_absolute_error(y_true, y_pred[:, 1:])*y_true[:,0]

class AB_Model(Gen_Model):
	def __init__(self, learning_rate, input_dim,  output_dim):
		Gen_Model.__init__(self, learning_rate, input_dim, output_dim)
		self.model = self._build_model()

	
	def get_model(self):
		return self.model
	def _build_model(self):

		main_input = Input(shape = self.input_dim, name = 'main_input', dtype = 'float32')

		X = Dense(10)(main_input)
		X = BatchNormalization()(X)
		X = LeakyReLU()(X)
		

		policy = Dense(self.output_dim, activation = 'softmax', name = "policy_head")(X)


		model = Model(inputs=[main_input], outputs=[policy])
		model.summary()
		model.compile(loss=custom_loss	,
			optimizer=SGD(lr=self.learning_rate))
		

		return model

