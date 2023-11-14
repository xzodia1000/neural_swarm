from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    @abstractmethod
    def evaluate(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass

    @abstractmethod
    def __str__(self):
        pass


class Sigmoid(Activation):
    def evaluate(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self.evaluate(x) * (1 - self.evaluate(x))

    def __str__(self):
        return "Sigmoid"


class Tanh(Activation):
    def evaluate(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def __str__(self):
        return "Tanh"


class Relu(Activation):
    def evaluate(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)

    def __str__(self):
        return "Relu"


class Softmax(Activation):
    def evaluate(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def derivative(self, x):
        return self.evaluate(x) * (1 - self.evaluate(x))

    def __str__(self):
        return "Softmax"


class Identity(Activation):
    def evaluate(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)

    def __str__(self):
        return "Identity"
