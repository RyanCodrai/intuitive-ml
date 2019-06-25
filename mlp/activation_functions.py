import numpy as np

class sigmoid():
	def __call__(self, x):
		return 1 / (1 + np.exp(-x))

	def derivative(self, x):
		return self.__call__(x) * (1 - self.__call__(x))

class relu():
	def __call__(self, x):
		return np.maximum(0, x)

	def derivative(self, x):
		return np.where(x <= 0, 0, 1)