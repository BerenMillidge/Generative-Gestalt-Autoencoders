import numpy as np
import keras
import sys

#Just a simple custom callback for terminating on NaN but when there are multiple losses

class TerminateOnNaNMultipleLosses(keras.callbacks.Callback):
	def _init__(self):
		super(TerminateOnNaNMultipleLosses,self).__init__()


	def on_batch_end(self, batch, logs=None):
		logs = logs or []
		losses = logs.get('loss')
		NaNDetected = False
		if losses is not None:
			for loss in losses:
				if np.isnan(loss) or np.isinf(loss):
					print('Batch %d: Invalid Loss %d, terminating training' %(batch) %(loss))
					NaNDetected = True

		if NaNDetected:
			self.model.stop_training = True

#another sinple callback to terminate if the loss increases more than a threshold
	
class TerminateOnIncreasedLoss(keras.callbacks.Callback):
	def __init_(self, diff = 1, monitor="val_loss"):
		super(TerminateOnIncreasedLoss, self).__init__()
		self.diff = diff
		self.monitor=monitor
		self.wait
		self.min_loss
		self.stopped_epoch

	def on_train_begin(self, logs=None):
		self.wait = 0
		self.stopped_epoch = 0
		self.min_loss = np.Inf

	def on_epoch_end(self, epoch, logs=None):
		current = logs.get(self.monitor)
		if current + diff > min_loss:
			self.stopped_epochs = epoch
			print('Loss increased: stopping training')
			self.model.stop_training= True	
		else: 
			self.stopped_epochs = epoch
			self.min_loss = current

	def on_train_end(self, logs=None):
		print('Stopped at epochs %d since loss increased' %(self.stopped_epoch))

		
		
