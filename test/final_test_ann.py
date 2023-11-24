import copy

import numpy as np
from tqdm import tqdm
from neural_swarm.ann.activation import Sigmoid
from neural_swarm.ann.ann import ANN
from neural_swarm.ann.loss import BinaryCrossEntropy
from neural_swarm.constants import RANDOM_INFORMANTS
from neural_swarm.pso.ann_function import ANNFunction
from neural_swarm.pso.pso import PSO
from test.test_data import prep_data


def test_case1():
    x, y_true = prep_data()

    ann = ANN()
    ann.add_layer(nodes=4, activation=Sigmoid())
    ann.add_layer(nodes=3, activation=Sigmoid())
    ann.add_layer(nodes=1, activation=Sigmoid())

    swarm_size = 37
    alpha, beta, gamma, delta = 0.4, 1.49, 1.49, 1.49
    epsilon = 0.4
    informants_type = RANDOM_INFORMANTS
    informants_size = 6
    iterations = 500

    acc = []

    for _ in tqdm(range(10)):
        ann_fun = ANNFunction(copy.deepcopy(ann), x, y_true, BinaryCrossEntropy(), True)

        pso = PSO(
            ann_fun,
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

        tmp_acc = 0
        for _, a in pso.evolve():
            tmp_acc = a
        acc.append(tmp_acc)

    print_final(acc, 1)


def test_case2():
    x, y_true = prep_data()

    ann = ANN()
    ann.add_layer(nodes=4, activation=Sigmoid())
    ann.add_layer(nodes=2, activation=Sigmoid())
    ann.add_layer(nodes=1, activation=Sigmoid())

    swarm_size = 37
    alpha, beta, gamma, delta = 0.4, 1.49, 1.49, 1.49
    epsilon = 0.4
    informants_type = RANDOM_INFORMANTS
    informants_size = 6
    iterations = 500

    acc = []

    for _ in tqdm(range(10)):
        ann_fun = ANNFunction(copy.deepcopy(ann), x, y_true, BinaryCrossEntropy(), True)

        pso = PSO(
            ann_fun,
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

        tmp_acc = 0
        for _, a in pso.evolve():
            tmp_acc = a
        acc.append(tmp_acc)

    print_final(acc, 2)


def test_case3():
    x, y_true = prep_data()

    ann = ANN()
    ann.add_layer(nodes=4, activation=Sigmoid())
    ann.add_layer(nodes=2, activation=Sigmoid())
    ann.add_layer(nodes=2, activation=Sigmoid())
    ann.add_layer(nodes=1, activation=Sigmoid())

    swarm_size = 37
    alpha, beta, gamma, delta = 0.4, 1.49, 1.49, 1.49
    epsilon = 0.4
    informants_type = RANDOM_INFORMANTS
    informants_size = 6
    iterations = 500

    acc = []

    for _ in tqdm(range(10)):
        ann_fun = ANNFunction(copy.deepcopy(ann), x, y_true, BinaryCrossEntropy(), True)

        pso = PSO(
            ann_fun,
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

        tmp_acc = 0
        for _, a in pso.evolve():
            tmp_acc = a
        acc.append(tmp_acc)

    print_final(acc, 3)


def print_final(acc, i):
    acc = np.array(acc)
    print("\n")
    print("Test Case: ", i)
    print("Max: ", np.max(acc))
    print("Average: ", np.average(acc))
    print("Std: ", np.std(acc))
    print("\n")
