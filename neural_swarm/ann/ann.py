import numpy as np
from neural_swarm.ann.layer import Layer
from neural_swarm.ann.network import Network


class ANN:
    def __init__(self):
        self.network = Network()
        self.dataset = None

    def add_layer(self, activation, nodes):
        layer = Layer(activation=activation, nodes=nodes)
        self.network.add_layer(layer)

    def run(self, x):
        return self.network.forward(x)

    def evaluate(self, x, y, loss):
        y = np.array(y)
        y_pred = self.network.forward(x)
        y_pred = np.array(y_pred).reshape(y.shape)

        ann_loss = loss.evaluate(y_pred, y)

        y_pred = (y_pred > 0.5).astype(int)
        correct_predictions = np.sum(y == y_pred)
        total_predictions = len(y)
        accuracy = correct_predictions / total_predictions

        return accuracy * 100, ann_loss
