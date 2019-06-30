class Sequential:
	def __init__(self):
		self.layers = []

	def add(self, new_layer):
		# If this is not the first layer being added to the neural network
		# (i.e the self.layers list is not empty)
		if self.layers:
			new_layer.previous_layer = self.layers[-1]
			self.layers[-1].next_layer = new_layer

		# Add the layer to the neural network
		self.layers.append(new_layer)




			