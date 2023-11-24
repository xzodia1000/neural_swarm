from abc import ABC, abstractmethod
import numpy as np


# Define an abstract base class for activation functions
class Activation(ABC):
    # Abstract method to evaluate the activation function
    @abstractmethod
    def evaluate(self, x):
        pass

    # Abstract method to calculate the derivative of the activation function
    @abstractmethod
    def derivative(self, x):
        pass

    # Abstract method to return a string representation of the activation function
    @abstractmethod
    def __str__(self):
        pass


# Sigmoid activation function class, inheriting from Activation
class Sigmoid(Activation):
    # Evaluate the Sigmoid function
    def evaluate(self, x):
        x = np.clip(x, -500, 500)  # Clip values to avoid numerical issues
        return 1 / (1 + np.exp(-x))

    # Calculate the derivative of the Sigmoid function
    def derivative(self, x):
        return self.evaluate(x) * (1 - self.evaluate(x))

    # Return the name of the activation function
    def __str__(self):
        return "Sigmoid"


# Tanh activation function class, inheriting from Activation
class Tanh(Activation):
    # Evaluate the Tanh function
    def evaluate(self, x):
        return np.tanh(x)

    # Calculate the derivative of the Tanh function
    def derivative(self, x):
        return 1 - np.tanh(x) ** 2

    # Return the name of the activation function
    def __str__(self):
        return "Tanh"


# ReLU (Rectified Linear Unit) activation function class, inheriting from Activation
class Relu(Activation):
    # Evaluate the ReLU function
    def evaluate(self, x):
        return np.maximum(0, x)

    # Calculate the derivative of the ReLU function
    def derivative(self, x):
        return np.where(x > 0, 1, 0)

    # Return the name of the activation function
    def __str__(self):
        return "Relu"


# Softmax activation function class, inheriting from Activation
class Softmax(Activation):
    # Evaluate the Softmax function
    def evaluate(self, x):
        e_x = np.exp(
            x - np.max(x, axis=-1, keepdims=True)
        )  # Shift values for numerical stability
        return e_x / e_x.sum(axis=-1, keepdims=True)

    # Calculate the derivative of the Softmax function
    # Note: This implementation may not be correct for all uses as softmax derivative is more complex
    def derivative(self, x):
        return self.evaluate(x) * (1 - self.evaluate(x))

    # Return the name of the activation function
    def __str__(self):
        return "Softmax"


# Identity activation function class, inheriting from Activation
class Identity(Activation):
    # Evaluate the Identity function (no change to input)
    def evaluate(self, x):
        return x

    # Derivative of the Identity function is 1
    def derivative(self, x):
        return np.ones_like(x)

    # Return the name of the activation function
    def __str__(self):
        return "Identity"
