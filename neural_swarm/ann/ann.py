import numpy as np
from .layer import Layer
from .network import Network


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

        accuracy = 0
        if loss.__str__() == "BinaryCrossEntropy":
            y_pred = (y_pred >= 0.5).astype(int)
            accuracy = np.mean(y_pred == y)
            ann_loss = loss.evaluate(y_pred, y)
        elif loss.__str__() == "CategoricalCrossEntropy":
            accuracy = self.categorical_classification_accuracy(y, y_pred)

        return accuracy * 100, ann_loss

    def binary_classification_accuracy(self, y_true, y_pred, threshold=0.5):
        y_pred = (y_pred > threshold).astype(int)
        return y_pred

    def categorical_classification_accuracy(self, y_true, y_pred):
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        return np.mean(y_pred == y_true)
