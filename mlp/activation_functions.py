import numpy as np

class sigmoid():
	def __call__(self, z):
		return 1 / (1 + np.exp(-z))

	def derivative(self, z):
		return self.__call__(z) * (1 - self.__call__(z))

class relu():
	def __call__(self, z):
		return np.maximum(0, z)

	def derivative(self, z):
		return np.greater(z, 0).astype(int)