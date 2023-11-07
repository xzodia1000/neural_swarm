from neural_swarm.ann.layer import Layer
from neural_swarm.ann.network import Network


class ANN:
    def __init__(self):
        self.network = Network()
        self.dataset = None

    def add_layer(self, activation, nodes, input=None, y_true=None):
        layer = Layer(activation=activation, nodes=nodes, input=input, y_true=y_true)
        self.network.add_layer(layer)
    
    def build(self):
        self.network.forward()
