import numpy as np
from neural_swarm.loss.loss import Loss


class Mse(Loss):
    def evaluate(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true)
