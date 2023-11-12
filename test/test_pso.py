from neural_swarm.ann.loss import BinaryCrossEntropy
from neural_swarm.pso.ann_function import ANNFunction
from neural_swarm.pso.constants import (
    LOCAL_GLOBAL_INFORMANTS,
    LOCAL_NEIGHBOUR_INFORMANTS,
    RANDOM_INFORMANTS,
)
from neural_swarm.pso.pso import PSO
from test.test_data import prep_data, prep_test_ann


def test_pso():
    ann = prep_test_ann()
    x, y_true = prep_data()
    loss = BinaryCrossEntropy()
    fun = ANNFunction(ann, x, y_true, loss)

    swarm_size = 15
    alpha = 1
    beta = 1
    gamma = 1
    delta = 1
    epsilon = 0.4
    informants_type = RANDOM_INFORMANTS
    informants_size = 3
    iterations = 1000

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

    result = pso.evolve()
    print(result)
