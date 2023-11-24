from abc import ABC, abstractmethod
import numpy as np
import copy


class Loss(ABC):
    """Abstract base class for loss functions."""

    @abstractmethod
    def evaluate(self, y_true, y_pred):
        """Computes the loss function given true labels and predicted labels."""
        pass

    @abstractmethod
    def derivative(self, y_true, y_pred):
        """Computes the derivative of the loss function with respect to the predicted labels."""
        pass

    @abstractmethod
    def __str__(self):
        """Returns a string representation of the loss function."""
        pass


class BinaryCrossEntropy(Loss):
    """Binary Cross Entropy loss, commonly used for binary classification tasks."""

    def evaluate(self, y_true, y_pred):
        """Calculates the binary cross entropy loss value."""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Clip predictions to avoid log(0)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def derivative(self, y_true, y_pred):
        """Computes the derivative of the binary cross entropy loss."""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y_true / y_pred) - ((1 - y_true) / (1 - y_pred))

    def __str__(self):
        """String representation of the binary cross entropy loss."""
        return "BinaryCrossEntropy"


class Mse(Loss):
    """Mean Squared Error loss, commonly used for regression tasks."""

    def evaluate(self, y_true, y_pred):
        """Calculates the MSE value."""
        return np.mean(np.square(y_true - y_pred))

    def derivative(self, y_true, y_pred):
        """Computes the derivative of the MSE."""
        return 2 * (y_pred - y_true)

    def __str__(self):
        """String representation of the MSE."""
        return "Mse"


class CategoricalCrossEntropy(Loss):
    """Categorical Cross Entropy loss, used for multi-class classification tasks."""

    def evaluate(self, y_true, y_pred):
        """Calculates the categorical cross entropy loss value."""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Clip predictions to avoid log(0)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    def derivative(self, y_true, y_pred):
        """Computes the derivative of the categorical cross entropy loss."""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return y_pred - y_true

    def __str__(self):
        """String representation of the categorical cross entropy loss."""
        return "CategoricalCrossEntropy"
