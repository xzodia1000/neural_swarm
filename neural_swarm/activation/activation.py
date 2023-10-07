from abc import ABC, abstractmethod


class Activation(ABC):
    @abstractmethod
    def evaluate(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass
