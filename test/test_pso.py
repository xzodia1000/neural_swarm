import numpy as np
from neural_swarm.ann.loss import BinaryCrossEntropy, CategoricalCrossEntropy
from neural_swarm.pso.ann_function import ANNFunction
from neural_swarm.constants import (
    LOCAL_GLOBAL_INFORMANTS,
    LOCAL_NEIGHBOUR_INFORMANTS,
    RANDOM_INFORMANTS,
)
from neural_swarm.pso.pso import PSO
from test.test_data import prep_data, prep_test_ann, prep_test_iris, prep_test_iris_ann


def test_iris_pso():
    ann = prep_test_iris_ann()
    x, y_true, _ = prep_test_iris()
    loss = CategoricalCrossEntropy()
    fun = ANNFunction(ann, x, y_true, loss, True)

    swarm_size = 15
    alpha = np.random.uniform(0, 1)
    beta, gamma, delta = np.random.dirichlet(np.ones(3)) * 4
    print("alpha: ", alpha, "beta: ", beta, "gamma: ", gamma, "delta: ", delta)
    epsilon = 0.4
    informants_type = RANDOM_INFORMANTS
    informants_size = 5
    iterations = 500

    pso = PSO(
        fun,
        swarm_size,
        alpha,
        beta,
        gamma,
        delta,
        epsilon,
        informants_type,
        informants_size,
        iterations,
        "MIN",
    )

    loss = []
    acc = []
    for l, a in pso.evolve():
        loss.append(l)
        acc.append(a)
    print("Accuracy: ", acc)
    print("\nLoss: ", loss)

    np.savetxt("test_outputs/pso_ann_iris_test.txt", ann.run(x), fmt="%s")

    with open("test_outputs/pso_ann_weights_iris_test.txt", "w") as f:
        for layer in ann.network.layers:
            f.write(str(layer.weights) + "\n")
            f.write(str(layer.activation) + "\n")

    print_final(
        loss[-1],
        acc[-1],
        "Iris",
        alpha,
        beta,
        gamma,
        delta,
        epsilon,
        informants_size,
        iterations,
    )


def test_pso():
    ann = prep_test_ann()
    x, y_true = prep_data()
    loss = BinaryCrossEntropy()
    fun = ANNFunction(ann, x, y_true, loss, True)

    swarm_size = 15
    alpha = np.random.uniform(0, 1)
    beta, gamma, delta = np.random.dirichlet(np.ones(3)) * 4
    print("alpha: ", alpha, "beta: ", beta, "gamma: ", gamma, "delta: ", delta)
    epsilon = 0.4
    informants_type = RANDOM_INFORMANTS
    informants_size = 5
    iterations = 500

    pso = PSO(
        fun,
        swarm_size,
        alpha,
        beta,
        gamma,
        delta,
        epsilon,
        informants_type,
        informants_size,
        iterations,
        "MIN",
    )

    loss = []
    acc = []
    for l, a in pso.evolve():
        loss.append(l)
        acc.append(a)
    print("Accuracy: ", acc)
    print("\nLoss: ", loss)

    np.savetxt("test_outputs/pso_ann_test.txt", ann.run(x), fmt="%s")

    with open("test_outputs/pso_ann_weights_test.txt", "w") as f:
        for layer in ann.network.layers:
            f.write(str(layer.weights) + "\n")
            f.write(str(layer.activation) + "\n")

    print_final(
        loss[-1],
        acc[-1],
        "Banknote",
        alpha,
        beta,
        gamma,
        delta,
        epsilon,
        informants_size,
        iterations,
    )


def print_final(
    loss, acc, dataset, alpha, beta, gamma, delta, epsilon, informants_size, iterations
):
    print("\n\n\n")
    print(
        "alpha: ",
        alpha,
        "beta: ",
        beta,
        "gamma: ",
        gamma,
        "delta: ",
        delta,
        "epsilon: ",
        epsilon,
        "informants_size: ",
        informants_size,
        "iterations: ",
        iterations,
    )
    print("Dataset: ", dataset)
    print("Loss: ", loss)
    print("Accuracy: ", acc)
    print("\n\n\n")
