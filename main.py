import numpy as np
from utils import *
from models import *
from keras.datasets import mnist
import matplotlib.pyplot as plt
from scipy.stats import norm

BATCH_SIZE = 64

def sanity_check_mnist(epochs=20, model=DCVAE, optimizer='sgd'):
	(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
	sh = xtrain.shape
	xtrain = xtrain.astype('float32')/255.
	xtrain = xtrain.reshape((xtrain.shape[0], sh[1], sh[2],1))
	xtest = xtest.astype('float32')/255.
	xtest = xtest.reshape((xtest.shape[0],sh[1],sh[2],1))
	print (" xtrain shape: " + str(xtrain.shape))
	print (" xtest shape: " + str(xtest.shape))

	vae, encoder, generator = model(xtrain.shape[1:])
	vae.fit(xtrain,xtrain, shuffle=True, epochs=epochs, batch_size=64, validation_data = (xtest, xtest))

		# so we want to display a 2d plot of the digit classes in the latent space
	x_test_encoded = encoder.predict(xtest, batch_size=batch_size)
	plt.figure(figsize=(6,6))
	plt.scatter(x_test_encoded[:,0], x_test_encoded[:,1],c=ytest)
	plt.colorbar()
	plt.show()

	# and now we want to display a 2d manifold of the digits
	"""
	n = 15 # we have a figure with 15x15 digits
	digit_size  = 28
	figure = np.zeros((digit_size*n, digit_size*n))
	#linearly spaced coordinates on the unit square are transformed through inverse CDF of gaussian
	# to produce values o fthe latent variables z sinec prior of latent space is gaussian
	grid_x = norm.ppf(np.linspace(0.05,0.95,n))
	grid_y = norm.ppf(np.linspace(0.05, 0.95,n))

	#I'm not really sure what ay of this does. hopefully it helps a bit thoguh!?
	for i, yi in enumerate(grid_x):
		for j, xi in enumerate(grid_y):
			z_sample = np.array([[xi, yi]])
			z_sample = np.tile(z_sample, batch_size).reshape(batch_size,2)
			x_decoded = generator.predict(z_sample, batch_size=batch_size)
			digit = x_decoded[0].reshape(digit_size, digit_size)
			figure[i*digit_size: (i+1) * digit_size,
					j*digit_size: (j+1)*digit_size] = digit
	plt.figure(figsize=(10,10))
	plt.imshow(figure, cmap='Greys_r')
	plt.show()
	"""
	N = 15
	for i in xrange(N):
		z_sample = np.random.multivariate_normal([0,0], [[1,0],[0,1]])
		z_sample = np.reshape(z_sample, (1,2))
		gen_img = generator.predict(z_sample)
		sh = gen_img.shape
		print gen_img.shape
		gen_img = np.reshape(gen_img, (sh[1], sh[2]))
		plt.imshow(gen_img, cmap='gray')
		plt.show() 

def test_gestalt_half_split_images(fname, epochs=20, model=DCVAE,optimizer='sgd', save_name=None):
	imgs = load_array(fname)
	imgs = imgs.astype('float32')/255.
	train, val, test = split_first_test_val_train(imgs)
	slicelefttrain, slicerighttrain = split_dataset_center_slice(train, 20)
	slicelefttest, slicerighttest = split_dataset_center_slice(test, 20)
	sliceleftval, slicerightval = split_dataset_center_slice(val, 20)
	#input_shape = slicelefttrain.shape[1:]

	#we do the concatenatoins to produce the full thing
	train1 = np.concatenate((slicelefttrain, slicerighttrain),axis=0)
	train2 = np.concatenate((slicerighttrain, slicelefttrain),axis=0)
	val1 = np.concatenate((sliceleftval, slicerightval), axis=0)
	val2 = np.concatenate((slicerightval, sliceleftval),axis=0)
	test1= np.concatenate((slicelefttest, slicerighttest), axis=0)
	test2 = np.concatenate((slicerighttest,slicelefttest),axis=0)

	#now we reshape as well
	sh = train1.shape
	#train1 = np.reshape(train1, (sh[0],sh[1],sh[2],1))
	#train2 = np.reshape(train2, (sh[0],sh[1],sh[2], 1))
	#val1 = np.reshape(val1, (len(val1),sh[1],sh[2],1))
	#val2 = np.reshape(val2, (len(val2),sh[1],sh[2], 1))
	#test1 = np.reshape(test1, (len(test1),sh[1],sh[2],1))
	#test2 = np.reshape(test2, (len(test2),sh[1],sh[2], 1))

	input_shape = train1.shape[1:]
	
	

	callbacks = build_callbacks("results/")

	vae, encoder,decoder = model(input_shape)
	#vae.compile(optimizer=optimizer,loss=None)
	#we fit the vae
	his = vae.fit(train1,train2, epochs=epochs, batch_size = BATCH_SIZE, shuffle=True, validation_data = (val1, val2), callbacks = callbacks)
	history = serialize_class_object(his)

	#now we try to get the predictions
	full_predictions = vae.predict(test1, batch_size=BATCH_SIZE)


	#encoder.compile(optimizer=optimizer, loss=None)
	#decoder.compile(optimizer=optimizer, loss=None)

	#I think we're oau with that then presumably
	if save_name:
		save_array([full_predictions, history],save_name)
	return full_predictions, history





if __name__ == '__main__':
	#test_gestalt_half_split_images("testimages_combined", epochs=1,save_name="results/	test_1")
	sanity_check_mnist(epochs=20)
	#test_gestalt_half_split_images("testimages_combined", epochs=500, save_name="results/test_1")
