class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

        if len(self.layers) >= 2:
            self.layers[-1].previous_layer = self.layers[-2]
            self.layers[-2].next_layer = self.layers[-1]
            self.layers[-2].set_weights()

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)

        return output
