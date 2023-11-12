from random import randrange
from neural_swarm.ann.activation import Sigmoid
from neural_swarm.ann.layer import Layer
from test.test_data import prep_data


def test_layer_init():
    layers = []
    activation = Sigmoid()

    # x, y = np.array([[2, 3, 2], [3, 3, 3]]), np.array([1, 0])
    x, y = prep_data()

    for i in range(10):
        if i == 0:
            layers.append(Layer(nodes=4, input=x))
        elif i == 9:
            layers.append(Layer(nodes=1, activation=activation, y_true=y))
        else:
            layers.append(Layer(nodes=randrange(1, 100), activation=activation))

        if i > 0:
            layers[i].previous_layer = layers[i - 1]
            layers[i - 1].next_layer = layers[i]

    test_epoch(layers)


def test_epoch(layers=None):
    if layers is None:
        return

    for i in range(10):
        print("\nEpoch " + str(i + 1))
        print("-" * 100 + "\n")
        test_forward(layers)
        test_backward(layers)


def test_forward(layers=None):
    if layers is None:
        return

    print("\nForward")
    print("-" * 100 + "\n")

    for i, layer in enumerate(layers):
        weighted_sum = layer.forward()

        print(f"Layer {i}")
        print("nodes = ", layer.nodes)
        print("weights = ", layer.weights)
        print("bias = ", layer.bias)
        print(weighted_sum)
        print("-" * 100 + "\n")


def test_backward(layers=None):
    if layers is None:
        return

    print("\nBackward")
    print("-" * 100 + "\n")
    for i, layer in enumerate(reversed(layers)):
        print(f"Layer {len(layers) - i}")
        print("nodes = ", layer.nodes)
        print("old weights = ", layer.weights)
        print("old bias = ", layer.bias)
        print("sigma = ", layer.weighted_sum)
        print()
        print("delta = ", layer.backward(0.001))
        print()
        print("new weights = ", layer.weights)
        print("new bias = ", layer.bias)
        print("-" * 100 + "\n")


test_layer_init()
