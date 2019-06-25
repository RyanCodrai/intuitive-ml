import numpy as np
from activation_functions import Relu, Sigmoid

class Dense():
	def __init__(self, output_shape, activation, input_shape=None):
		self.output_shape = output_shape

		if input_shape is not None:
			self.input_shape = input_shape

		# Instantiate the weights and bias
		self.weights = None
		self.bias = None

		# Instantiate the chosen activation function
		if activation is "relu":
			self.activation = Relu()
		if activation is "sigmoid":
			self.activation = Sigmoid()





		