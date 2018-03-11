from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.models import load_model
from keras.datasets import cifar10
from keras import optimizers
from utils import *

img_rows, img_cols, img_chns = 28, 28, 1
if K.image_data_format() == 'channels_first':
		original_img_size = (img_chns, img_rows, img_cols)
else:
		original_img_size = (img_rows, img_cols, img_chns)

epochs = 50
batch_size = 100
# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3

	
latent_dim = 2
intermediate_dim = 128
epsilon_std = 1.0
activation = 'relu'


def vae_model(input_shape,epochs, batch_size, filters, num_conv, latent_dim, intermediate_dim, epsilon_std, activation='relu', save=True, save_fname="results/gestalt_VAE"):
	# input image dimensions
	rows, cols, channels = input_shape


	x = Input(shape=input_shape)
	conv_1 = Conv2D(img_chns,
		            kernel_size=(2, 2),
		            padding='same', activation=activation)(x)
	conv_2 = Conv2D(filters,
		            kernel_size=(2, 2),
		            padding='same', activation=activation,
		            strides=(2, 2))(conv_1)
	conv_3 = Conv2D(filters,
		            kernel_size=num_conv,
		            padding='same', activation=activation,
		            strides=1)(conv_2)
	conv_4 = Conv2D(filters,
		            kernel_size=num_conv,
		            padding='same', activation=activation,
		            strides=2)(conv_3)
	flat = Flatten()(conv_4)
	hidden = Dense(intermediate_dim, activation=activation)(flat)

	z_mean = Dense(latent_dim)(hidden)
	z_log_var = Dense(latent_dim)(hidden)


	def sampling(args):
		z_mean, z_log_var = args
		epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
		                          mean=0., stddev=epsilon_std)
		return z_mean + K.exp(z_log_var) * epsilon

	# note that "output_shape" isn't necessary with the TensorFlow backend
	# so you could write `Lambda(sampling)([z_mean, z_log_var])`
	z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

	# we instantiate these layers separately so as to reuse them later
	decoder_hid = Dense(intermediate_dim, activation=activation)
	#decoder_upsample = Dense(filters * 14 * 14, activation=activation)
	decoder_upsample = Dense(filters * rows/4 * cols/4, activation=activation)

	if K.image_data_format() == 'channels_first':
		output_shape = (batch_size, filters, rows/4, cols/4)
	else:
		output_shape = (batch_size, rows/4, cols/4, filters)

	decoder_reshape = Reshape(output_shape[1:])
	decoder_deconv_1 = Conv2DTranspose(filters,
		                               kernel_size=num_conv,
		                               padding='same',
		                               strides=1,
		                               activation=activation)
	decoder_deconv_2 = Conv2DTranspose(filters,
		                               kernel_size=num_conv,
		                               padding='same',
		                               strides=2,
		                               activation=activation)
	if K.image_data_format() == 'channels_first':
		output_shape = (batch_size, filters, rows+1, cols+1)
	else:
		output_shape = (batch_size, rows+1, cols+1, filters)
	decoder_deconv_3_upsamp = Conv2DTranspose(filters,
		                                      kernel_size=(3, 3),
		                                      strides=(2, 2),
		                                      padding='valid',
		                                      activation=activation)
	decoder_mean_squash = Conv2D(channels,
		                         kernel_size=2,
		                         padding='valid',
		                         activation='sigmoid')

	hid_decoded = decoder_hid(z)
	up_decoded = decoder_upsample(hid_decoded)
	reshape_decoded = decoder_reshape(up_decoded)
	deconv_1_decoded = decoder_deconv_1(reshape_decoded)
	deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
	x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
	x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

	#y = Input(shape=input_shape)

	# instantiate VAE model
	vae = Model(x, x_decoded_mean_squash)

	# Compute VAE loss
	#xent_loss = img_rows * img_cols * metrics.binary_crossentropy(
	#	K.flatten(x),
	#	K.flatten(x_decoded_mean_squash))
	#kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
	#xent_loss = reconstruction_loss(rows, cols, x, x_decoded_mean_squash)
	kl = kl_loss(z_mean, z_log_var)
	#vae_loss = K.mean(xent_loss + kl)
	vae.add_loss(kl)

	# build a model to project inputs on the latent space
	encoder = Model(x, z_mean)

	#generator
	# build a digit generator that can sample from the learned distribution
	decoder_input = Input(shape=(latent_dim,))
	_hid_decoded = decoder_hid(decoder_input)
	_up_decoded = decoder_upsample(_hid_decoded)
	_reshape_decoded = decoder_reshape(_up_decoded)
	_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
	_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
	_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
	_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
	generator = Model(decoder_input, _x_decoded_mean_squash)

	#implement model saving simply here
	if save:
		vae.save(save_fname + "_VAE")
		encoder.save(save_fname+ "_encoder")
		generator.save(save_fname + "_generator")

	return vae, encoder, generator, z_mean, z_log_var

#split out the losses into functions. This seems to have worked so far!
def reconstruction_loss(y, x_decoded):
	#let's hard code this for now
	rows = 8
	cols = 32
	rec_loss = rows * cols * metrics.binary_crossentropy(K.flatten(y), K.flatten(x_decoded))
	print("Rec loss: " + str(rec_loss))
	return rec_loss
def unnormalised_reconstruction_loss(x_decoded, y):
	rec_loss = metrics.binary_crossentropy(K.flatten(x_decoded), K.flatten(y))
	print("Rec loss: " + str(rec_loss))
	return rec_loss

def kl_loss(z_mean, z_log_var):
	klloss =  -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
	print("KL loss: " + str(klloss))
	return klloss

def predict_display(N, testslices, actuals,generator):
	testsh = testslices.shape
	actualsh = actuals.shape
	#if len(testsh) ==3:
	testslices = np.reshape(testslices,(testsh[0], testsh[1], testsh[2]))
	actuals = np.reshape(actuals, (actualsh[0], actualsh[1], actualsh[2]))

	#epsilon_std = 1.0
	for i in xrange(N):
		#epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
		                          #mean=0., stddev=epsilon_std)
		#epsilon = np.random.multivariate_normal(z_mean, np.exp(z_log_var))
		#print("Z MEAN")
		##print(z_mean)
		#print("Z LOG VAR")
		#print(z_log_var)
		#z =  z_mean + K.exp(z_log_var) * epsilon
		#print(z)
		##try evaling it
		#z = K.eval(z)
		#print(z)
		z = np.random.multivariate_normal([0,0],[[1,0],[0,1]])
		z = np.reshape(z, (1,2))
		pred = generator.predict(z,batch_size=1)
		sh = pred.shape
		pred = np.reshape(pred, (sh[1], sh[2],sh[3]))
		if sh[3] ==1:
			pred = np.reshape(pred, (sh[1],sh[2]))
		print("PRED")
		print(pred.shape)
		print("testslice")
		print(testslices[i].shape)
		print("actual")
		print(actuals[i].shape)
		plot_three_image_comparison(testslices[i], pred, actuals[i], reshape=False)

def mnist_experiment():
	(x_train, _), (x_test, y_test) = mnist.load_data()
	x_train = x_train.astype('float32') / 255.
	x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
	x_test = x_test.astype('float32') / 255.
	x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

	lefttrain, righttrain = split_dataset_center_slice(x_train, 12)
	lefttest, righttest = split_dataset_center_slice(x_test, 12)
	#print(lefttrain.shape
	#imgs= load_array("testimages_combined")
	#imgs = imgs.astype('float32')/255. # normalise here. This might solve some issues
	#print(imgs.shape)
	#x_train, x_test = split_first_test_train(imgs)
	#print('x_train.shape:', x_train.shape)
	shape = lefttrain.shape[1:]
	#shape=lefttrain.shape[1:]

	vae, encoder, generator, z_mean, z_log_var = vae_model(shape,epochs, batch_size, filters, num_conv, latent_dim, intermediate_dim, epsilon_std)
	vae.compile(optimizer='adam',loss=reconstruction_loss)
	vae.summary()

	callbacks = build_callbacks("results/callbacks/")

	his = vae.fit(lefttrain,righttrain,
		shuffle=True,epochs=epochs, batch_size=batch_size,
		validation_data=(lefttest, righttest))
	history = serialize_class_object(his)
	save_array(history, "results/VAE_train_history_2")
	#save models
	vae.save('results/VAE_vae_model_1')
	generator.save('results/VAE_generator_model_1')
	encoder.save('results/VAE_encoder_model_1')


	# display a 2D plot of the digit classes in the latent space
	x_test_encoded = encoder.predict(lefttest, batch_size=batch_size)
	plt.figure(figsize=(6, 6))
	plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1])
	#plt.colorbar()
	plt.show()

	# build a digit generator that can sample from the learned distribution
	#decoder_input = Input(shape=(latent_dim,))
	#_hid_decoded = decoder_hid(decoder_input)
	#_up_decoded = decoder_upsample(_hid_decoded)
	#_reshape_decoded = decoder_reshape(_up_decoded)
	#_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
	#_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
	#_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
	#_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
	#generator = Model(decoder_input, _x_decoded_mean_squash)

	preds = vae.predict(lefttest)
	save_array(preds, "results/mnist_vae_preds_2")
			
	predict_display(20, lefttest, x_test, generator)
	##Tensor("add_1:0", shape=(?, 2), dtype=float32)


	# display a 2D manifold of the digits
	n = 15  # figure with 15x15 digits
	digit_width = shape[0]
	digit_height = shape[1]
	figure = np.zeros((digit_width * n, digit_height * n))

	grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
	grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

	for i, yi in enumerate(grid_x):
		for j, xi in enumerate(grid_y):
		    z_sample = np.array([[xi, yi]])
		    z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
		   # print(z_sample)
		    x_decoded = generator.predict(z_sample, batch_size=batch_size)
		    digit = x_decoded[0,:,:,0].reshape(digit_width, digit_height)
		    figure[i * digit_width: (i + 1) * digit_width,
		           j * digit_height: (j + 1) * digit_height] = digit

	plt.figure(figsize=(10, 10))
	plt.imshow(figure, cmap='Greys_r')
	plt.show()


def cifar10_experiment():
	# so cifar isn't workingat all. but mnist does, if I'm not mistaken. We've got to figure out therefor ,what is wrong with the CIFAR code
	slice_width = 8
	epochs=20
	(xtrain, ytrain),(xtest, ytest) = cifar10.load_data()
	xtrain = xtrain.astype('float32')/255.
	xtest = xtest.astype('float32')/255.
	sh = xtrain.shape
	#let's reshape to only be 2d like mnist. that could be causing the issues
	# yes! that's working a little better. except no. it's just getting stuck at 0.6933 vs mnist. I'm not sure why this is the case, but it's infuriating as it's not working at all
	#could be an issue with the learning rate?
	xtrain = np.reshape(xtrain[:,:,:,0],(len(xtrain), sh[1], sh[2],1))
	xtest = np.reshape(xtest[:,:,:,0], (len(xtest), sh[1],sh[2],1))

	lefttrain, righttrain = split_dataset_center_slice(xtrain, slice_width)
	lefttest, righttest = split_dataset_center_slice(xtest, slice_width)
	#print(lefttrain.shape
	#imgs= load_array("testimages_combined")
	#imgs = imgs.astype('float32')/255. # normalise here. This might solve some issues
	#print(imgs.shape)
	#x_train, x_test = split_first_test_train(imgs)
	#print('x_train.shape:', x_train.shape)
	shape = lefttrain.shape[1:]
	#shape=lefttrain.shape[1:]

	vae, encoder, generator, z_mean, z_log_var = vae_model(shape,epochs, batch_size, filters, num_conv, latent_dim, intermediate_dim, epsilon_std)
	

	#define optimisers here
	learning_rate = 0.00001
	sgd_decay = 1e-6
	sgd_momentum=0.9
	nesterov=True
	sgd = optimizers.SGD(lr = learning_rate, decay=sgd_decay, momentum=sgd_momentum, nesterov=nesterov)

	adam_lr = 0.001
	adam_beta_1 = 0.9
	adam_beta_2=0.999
	adam = optimizers.Adam(lr=adam_lr, beta_1=adam_beta_1, beta_2=adam_beta_2)


	vae.compile(optimizer=adam,loss=reconstruction_loss)
	vae.summary()

	callbacks = build_callbacks("results/callbacks/")

	his = vae.fit(lefttrain,righttrain,
		shuffle=True,epochs=epochs, batch_size=batch_size,
		validation_data=(lefttest, righttest))

	history = serialize_class_object(his)
	save_array(history, "results/VAE_train_history_cifar_3")
	#save models
	vae.save('results/VAE_vae_model_1_cifar')
	generator.save('results/VAE_generator_model_3_cifar')
	encoder.save('results/VAE_encoder_model_3_cifar')


	# display a 2D plot of the digit classes in the latent space
	x_test_encoded = encoder.predict(lefttest, batch_size=batch_size)
	plt.figure(figsize=(6, 6))
	plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1])
	#plt.colorbar()
	plt.show()

	# build a digit generator that can sample from the learned distribution
	#decoder_input = Input(shape=(latent_dim,))
	#_hid_decoded = decoder_hid(decoder_input)
	#_up_decoded = decoder_upsample(_hid_decoded)
	#_reshape_decoded = decoder_reshape(_up_decoded)
	#_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
	#_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
	#_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
	#_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
	#generator = Model(decoder_input, _x_decoded_mean_squash)

	preds = vae.predict(lefttest)
	save_array(preds, "results/cifar_vae_preds_2")
			
	predict_display(20, lefttest, righttest, generator)
	##Tensor("add_1:0", shape=(?, 2), dtype=float32)


	# display a 2D manifold of the digits
	n = 15  # figure with 15x15 digits
	digit_width = shape[0]
	digit_height = shape[1]
	figure = np.zeros((digit_width * n, digit_height * n))
	grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
	grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

	for i, yi in enumerate(grid_x):
		for j, xi in enumerate(grid_y):
		    z_sample = np.array([[xi, yi]])
		    z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
		   # print(z_sample)
		    x_decoded = generator.predict(z_sample, batch_size=batch_size)
		    digit = x_decoded[0,:,:,0].reshape(digit_width, digit_height)
		    figure[i * digit_width: (i + 1) * digit_width,
		           j * digit_height: (j + 1) * digit_height] = digit

	plt.figure(figsize=(10, 10))
	plt.imshow(figure, cmap='Greys_r')
	plt.show()



	"""
	print(xtrain.shape)
	lefttrain, righttrain = split_dataset_center_slice(xtrain, slice_width)
	lefttest, righttest = split_dataset_center_slice(xtest, slice_width)
	print(lefttrain.shape)
	#let's print them to make sure we're doing okay
	
	for i in xrange(20):
		fig = plt.figure()
		print(lefttrain[i].shape)
		l = np.reshape(lefttrain[i], (32,12))
		r = np.reshape(righttrain[i], (32,12))
		ax1 = fig.add_subplot(121)
		plt.imshow(l)
		ax2 = fig.add_subplot(122)
		plt.imshow(r)
		plt.show(fig)
	
	
	shape = lefttrain.shape[1:]

	vae, encoder, generator, z_mean, z_log_var = vae_model(shape,epochs, batch_size, filters, num_conv, latent_dim, intermediate_dim, epsilon_std,save_fname="results/vae_cifar_model")
	vae.compile(optimizer='sgd',loss=unnormalised_reconstruction_loss)
	vae.summary()

	callbacks = build_callbacks("results/callbacks/")

	his = vae.fit(lefttrain,lefttrain,
		shuffle=True,epochs=epochs, batch_size=batch_size,
		validation_data=(lefttest, righttest))

	#just for quick tests of the thing
	x_test = lefttest

	history = serialize_class_object(his)
	save_array(history, "results/VAE_train_history_cifar")


	# display a 2D plot of the digit classes in the latent space
	x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
	plt.figure(figsize=(6, 6))
	plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1])
	#plt.colorbar()
	plt.show()

	preds = vae.predict(x_test)
	save_array(preds, "results/cifar_vae_preds")
			
	predict_display(20, lefttest, x_test, generator)
	"""


if __name__ == '__main__':
	cifar10_experiment()
	#mnist_experiment()
