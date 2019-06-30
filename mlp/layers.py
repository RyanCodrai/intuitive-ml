import numpy as np
from activation_functions import Relu, Sigmoid

class Dense():
	def __init__(self, units, activation_function):
		self.units = units

		# Instantiate the weights and bias
		self.weights = None
		self.bias = None

		# Instantiate the chosen activation function
		if activation_function is "relu":
			self.activation_function = Relu()
		if activation_function is "sigmoid":
			self.activation_function = Sigmoid()

	def forward_pass(self, X):
		# Calculate the affine transformation, Z
		self.Z = np.dot(self.W, X) + self.b
		# Calculate the activations, A
		self.A = self.activation_function(self.Z)
		return self.A

class Input():
	def __init__(self, units):
		self.units = units
		self.A = None

	def forward_pass(self, X):
		return X




		