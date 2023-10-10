class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

        if len(self.layers) > 1:
            self.layers[-1].previous_layer = self.layers[-2]
            self.layers[-2].next_layer = self.layers[-1]

    def forward(self):
        for layer in self.layers:
            layer.forward()

    def backward(self, learning_rate):
        for layer in reversed(self.layers):
            layer.backward(learning_rate)
