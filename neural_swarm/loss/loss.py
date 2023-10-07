from abc import ABC, abstractmethod


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
