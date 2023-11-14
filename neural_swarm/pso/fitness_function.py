from abc import ABC, abstractmethod


class FitnessFunction:
    """Abstract class for fitness functions."""

    def __init__(self):
        pass

    def get_dimension(self):
        """Returns the dimension of the fitness function."""
        pass

    def set_variable(self, particle):
        """Sets the variable of the fitness function."""
        pass

    @abstractmethod
    def evaluate(self, particle):
        """Evaluates the particle's fitness."""
        pass
