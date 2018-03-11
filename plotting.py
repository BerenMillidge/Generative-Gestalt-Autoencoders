import numpy as np
import matplotlib.pyplot as plt
from utils import *

def plot_slices_predictions_comparison(data_fname, predictions_fname, N = 20, ret=False):
	leftslice, rightslice = load_array(data_fname)
	predictions = load_array(predictions_fname)[0]
	assert leftslice.shape == rightslice.shape == predictions.shape,'all data should be of the same shape'
	figs = []
	
	for i in xrange(N):
		fig = plt.figure(figsize=10)
		ax1 = fig.add_subplot(131)
		plt.imshow(leftslice[i])
		plt.title('Left Slice')
		plt.xticks=([])
		plt.yticks=([])

		ax2 = fig.add_subplot(132)
		plt.imshow(rightslice[i])
		plt.title('Right Slice')
		plt.xticks=([])
		plt.yticks=([])
		
		ax1 = fig.add_subplot(133)
		plt.imshow(prediction[i])
		plt.title('Predicted Right Slice')
		plt.xticks=([])
		plt.yticks=([])

		plt.tight_layout()
		plt.show(fig)
		if ret:
			figs.append(fig)
		
	if ret:
		return figs
