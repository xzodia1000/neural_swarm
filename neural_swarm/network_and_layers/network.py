class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self):
        for layer in self.layers:
            layer.forward()

    def backward(self, learning_rate):
        for layer in reversed(self.layers):
            layer.backward(learning_rate)
