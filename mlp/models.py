class Sequential:
	def __init__(self):
		self.layers = []

	def add(self, layer):
		# If this is the first layer being added to the neural network (i.e the self.layers list is empty)
		# and no input_shape has been provided then produce an error as we cannot infer the input shape
		if self.layers is None and layer.input_shape is None:
			raise Exception('The first layer in a Sequential model must get an `input_shape`')

		# Add the layer to the neural network
		self.layers.append(layer)



			