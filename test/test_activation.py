import numpy as np
from neural_swarm.ann.activation import Sigmoid, Tanh, Relu


def test_logistic():
    activation = Sigmoid()
    assert np.isclose(activation.evaluate(0), 0.5)
    assert np.isclose(activation.evaluate(1), 0.73105858)

    assert np.isclose(activation.derivative(0), 0.25)


def test_tanh():
    activation = Tanh()
    assert np.isclose(activation.evaluate(0), 0)
    assert np.isclose(activation.evaluate(np.pi), np.tanh(np.pi))

    assert np.isclose(activation.derivative(0), 1)


def test_relu():
    activation = Relu()
    assert np.isclose(activation.evaluate(0), 0)
    assert np.isclose(activation.evaluate(5), 5)
    assert np.isclose(activation.evaluate(-5), 0)

    assert np.isclose(activation.derivative(0), 0)
    assert np.isclose(activation.derivative(5), 1)
    assert np.isclose(activation.derivative(-5), 0)
