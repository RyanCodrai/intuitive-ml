import numpy as np

class sigmoid():
	def __call__(self, z):
		return 1 / (1 + np.exp(-z))

	def derivative(self, dA, Z):
		sig = sigmoid()
		sig = sig(Z)
		return dA * sig * (1 - sig)

class relu():
	def __call__(self, z):
		return np.maximum(0, z)

	def derivative(self, dA, Z):
		dZ = np.array(dA, copy = True)
		dZ[Z <= 0] = 0;
		return dZ;