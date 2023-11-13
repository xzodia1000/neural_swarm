import numpy as np
from neural_swarm.ann.activation import Sigmoid, Softmax, Tanh, Relu


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


def test_softmax():
    activation = Softmax()

    input1 = np.array([1, 2, 3])
    output1 = activation.evaluate(input1)
    assert np.allclose(output1.sum(), 1), "Softmax probabilities should sum to 1"

    # Test case 2: 2D input
    input2 = np.array([[1, 2, 3], [1, 2, 3]])
    output2 = activation.evaluate(input2)
    assert np.allclose(
        output2.sum(axis=1), [1, 1]
    ), "Each row of softmax probabilities should sum to 1"

    # Test case 3: Numerical stability with large numbers
    input3 = np.array([1000, 2000, 3000])
    output3 = activation.evaluate(input3)
    assert not np.any(
        np.isnan(output3)
    ), "Softmax should not return NaN for large numbers"

    # Test case 4: Numerical stability with small numbers
    input4 = np.array([-1000, -2000, -3000])
    output4 = activation.evaluate(input4)
    assert not np.any(
        np.isnan(output4)
    ), "Softmax should not return NaN for small numbers"

    # Test case 5: Check if softmax is invariant to constant offsets
    input5 = np.array([1, 2, 3])
    offset = 1000
    output5 = activation.evaluate(input5)
    output5_offset = activation.evaluate(input5 + offset)
    assert np.allclose(
        output5, output5_offset
    ), "Softmax should be invariant to constant offsets"
