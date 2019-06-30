import numpy as np
from activation_functions import Relu, Sigmoid

class Layer():
	def __init__(self):
		self.next_layer = None
		self.previous_layer = None


class Dense(Layer):
	def __init__(self, shape):
		Layer.__init__(self)
		self.shape = shape


	def forward_propagation(self, X):
		# Save the input for the layer
		self.X = X
		# Calculate the affine transformation for the layer
		Z = np.dot(self.W, X) + self.b

		# If there is a subsequent layer then return it's output, else return the output of this layer
		if self.next_layer is not None:
			return self.next_layer.forward_propagation(Z)
		else:
			return Z

	def backward_propogation(self, error_signal):
		# number of examples
		m = self.X.shape[1]
		
		# derivative of the matrix W
		self.dW = np.dot(error_signal, self.X.T) / m
		# derivative of the vector b
		self.db = np.sum(error_signal, axis=1, keepdims=True) / m

		if self.previous_layer is not None:
			self.previous_layer.backward_propogation(np.dot(self.W.T, error_signal))

	def initalise(self):
		# Let units(l) be the number of units for layer l
		# There are units(l-1) weights for every unit in the current layer
		# This is why the weight matrix is units(l) * units(l - 1) in size
		self.W = np.random.randn(self.shape, self.previous_layer.shape) * 0.1
		# There is one bias value for each unit in the current layer
		self.b = np.random.randn(self.shape, 1) * 0.1


class Activation(Layer):
	def __init__(self, activation_function):
		Layer.__init__(self)

		# Instantiate the chosen activation function
		if activation_function is "relu":
			self.activation_function = Relu()
		if activation_function is "sigmoid":
			self.activation_function = Sigmoid()

	def forward_propagation(self, X):
		# Save the input for the layer
		self.X = X
		A = self.activation_function(self.X)

		# If there is a subsequent layer then return it's output, else return the output of this layer
		if self.next_layer is not None:
			return self.next_layer.forward_propagation(A)
		else:
			return A

	def backward_propogation(self, error_signal):
		# If there is a preceding layer then pass the error_signal
		if self.previous_layer is not None:
			self.previous_layer.backward_propogation(error_signal * self.activation_function.derivative(self.X))

	def initalise(self):
		self.shape = self.previous_layer.shape


class Input(Layer):
	def __init__(self, shape):
		Layer.__init__(self)
		self.shape = shape

	def forward_propagation(self, X):
		self.X = X

		# If there is a subsequent layer then return it's output, else return the output of this layer
		if self.next_layer is not None:
			return self.next_layer.forward_propagation(X)
		else:
			return X

	def backward_propogation(self, error_signal):
		pass




		