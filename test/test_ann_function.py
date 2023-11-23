import numpy as np
from neural_swarm.pso.ann_function import ANNFunction
from test.test_data import prep_test_ann


def test_get_dimension():
    ann = prep_test_ann()
    fun = ANNFunction(ann, None, None, None)
    dimension = fun.get_dimension()
    assert dimension == 17


def test_set_variable():
    ann = prep_test_ann()
    fun = ANNFunction(ann, None, None, None)
    dimension = fun.get_dimension()
    decoded = np.random.uniform(-1, 1, dimension)
    print(fun.set_variable(decoded))
