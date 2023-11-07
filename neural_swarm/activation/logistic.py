import numpy as np
from neural_swarm.activation.activation import Activation


class Logistic(Activation):
    def evaluate(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self.evaluate(x) * (1 - self.evaluate(x))

    def __str__(self):
        return "Logistic"
