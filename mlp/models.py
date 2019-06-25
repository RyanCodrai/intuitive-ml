class Sequential:
	def __init__(self):
		self.layers = []


	def add(self, layer):
		# Check if first layer added has input shape
		if self.layers is None and layer.input_shape is None:
			raise Exception('The first layer in a Sequential model must get an `input_shape`')

		if self.layers:
			layer.input_shape = self.layers[-1].units

		self.layers.append(layer)



			