import numpy as np 
import tensorflow as tf 
import data_reader 

class NNExample:

	def __init__(self,
		learning_rate=0.015,
		training_iteration=30,
		batch_size=50,
		hidden_layer_n=255):


		self.dataset = data_reader.MNIST()


		self.learning_rate 		= learning_rate
		self.training_iteration = training_iteration
		self.batch_size 		= batch_size
		input_layer_n 			= self.dataset.input_dim
		output_layer_n 			= self.dataset.output_dim
		self.display_step		= 3


		#todo

	def run(self):
		#todo 
		
if __name__== "__main__":

	nn = NNExample()
	nn.run()
