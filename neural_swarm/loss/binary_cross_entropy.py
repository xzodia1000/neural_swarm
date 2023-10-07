import numpy as np
from neural_swarm.loss.loss import Loss


class BinaryCrossEntropy(Loss):
    def evaluate(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        term_a = y_true * np.log(y_pred)
        term_b = (1 - y_true) * np.log(1 - y_pred)

        return -np.mean(term_a + term_b)

    def derivative(self, y_true, y_pred):
        return (y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
