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

    @abstractmethod
    def __str__(self):
        pass


class Hinge(Loss):
    def evaluate(self, y_true, y_pred):
        actual_copy = copy.deepcopy(y_true)
        actual_copy[actual_copy == 0] = -1
        return np.mean(np.maximum(0, 1 - actual_copy * y_pred))

    def derivative(self, y_true, y_pred):
        return np.where(1 - y_true * y_pred > 0, -y_true, 0)

    def __str__(self):
        return "Hinge"


class BinaryCrossEntropy(Loss):
    def evaluate(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def derivative(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y_true / y_pred) - ((1 - y_true) / (1 - y_pred))

    def __str__(self):
        return "BinaryCrossEntropy"


class Mse(Loss):
    def evaluate(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true)

    def __str__(self):
        return "Mse"


class CategoricalCrossEntropy(Loss):
    def evaluate(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    def derivative(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return y_pred - y_true

    def __str__(self):
        return "CategoricalCrossEntropy"
