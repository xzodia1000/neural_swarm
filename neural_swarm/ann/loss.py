from abc import ABC, abstractmethod
import numpy as np
import copy


class Loss(ABC):
    """Abstract class for loss functions."""

    @abstractmethod
    def evaluate(self, y_true, y_pred):
        """Computes the loss function."""
        pass

    @abstractmethod
    def derivative(self, y_true, y_pred):
        """Computes the derivative of the loss function."""
        pass


class Hinge(Loss):
    def evaluate(self, y_true, y_pred):
        actual_copy = copy.deepcopy(y_true)
        actual_copy[actual_copy == 0] = -1
        return np.mean(np.maximum(0, 1 - actual_copy * y_pred))

    def derivative(self, y_true, y_pred):
        return np.where(1 - y_true * y_pred > 0, -y_true, 0)


class BinaryCrossEntropy(Loss):
    def evaluate(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        term_a = y_true * np.log(y_pred)
        term_b = (1 - y_true) * np.log(1 - y_pred)

        return -np.mean(term_a + term_b)

    def derivative(self, y_true, y_pred):
        return (y_true / y_pred) + ((1 - y_true) / (1 - y_pred))


class Mse(Loss):
    def evaluate(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true)
