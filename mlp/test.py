import numpy as np
import time

class sigstat(object):
	def __new__(self, z):
		#ignore the args 
		return 1 / (1 + np.exp(-z))

	@staticmethod
	def derivative(z):
		return (1/(1+np.exp(z))) * (1 - (1/(1+np.exp(z))))



class sigclass():
	def __call__(self, z):
		return 1 / (1 + np.exp(-z))

	def derivative(self, z):
		return self.__call__(z) * (1 - self.__call__(z))


start = time.time()

for x in range(1000000):

	sigstat.derivative(0.5)
	sigstat(0.5)

print(time.time() - start)

start = time.time()

sig = sigclass()

for x in range(1000000):

	sig.derivative(0.5)
	sig(0.5)

print(time.time() - start)