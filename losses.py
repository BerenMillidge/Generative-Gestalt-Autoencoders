import numpy as np
import keras
from keras import backend as K
from keras.layers import Layer
from keras import metrics




def vae_loss(y_true, y_pred):
	recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
	kl= 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)
	return recon + kl

def metrics_flat_binary_crossentropy_loss(x_mean_squash, y):
	y = K.flatten(y)
	xent_loss = metrics.binary_crossentropy(x_mean_squash, y)
	kl_loss = -0.5 * K.mean(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
	return K.mean(xent_loss + kl_loss)

class CustomVariationalLayer(Layer):
	def __init__(self,z_mean, z_log_var, **kwargs):
		self.is_placeholder=True
		self.z_mean = z_mean
		self.z_log_var = z_log_var
		super(CustomVariationalLayer, self).__init__(**kwargs)

	def vae_loss(self, x, x_decoded_mean_squash):
		x = K.flatten(x)
		x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
		#xent_loss = img_rows*img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
		xent_loss = metrics.binary_crossentropy(x, x_decoded_mean_squash)		
		kl_loss = -0.5 * K.mean(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
		return K.mean(xent_loss + kl_loss)

	def call(self, inputs):
		x = inputs[0]
		x_decoded_mean_squash = inputs[1]
		loss = self.vae_loss(x, x_decoded_mean_squash)
		self.add_loss(loss, inputs=inputs)
		return x

