import numpy as np
from neural_swarm.loss.loss import Loss


class Hinge(Loss):
    def evaluate(self, y_true, y_pred):
        return np.mean(np.maximum(0, 1 - y_true * y_pred))

    def derivative(self, y_true, y_pred):
        return np.where(1 - y_true * y_pred > 0, -y_true, 0)
