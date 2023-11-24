import numpy as np
from sklearn.metrics import mean_squared_error, log_loss
from neural_swarm.ann.loss import (
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    Mse,
)
from test.test_data import (
    get_rand_iris_predictions,
    get_rand_predictions,
    get_randn_predictions,
)


def test_binary_cross_entropy():
    y_true, y_pred = get_rand_predictions()
    loss = BinaryCrossEntropy()
    assert loss.evaluate(y_true, y_pred) == log_loss(y_true, y_pred)


def test_mse():
    y_true, y_pred = get_randn_predictions()
    loss = Mse()
    assert loss.evaluate(y_true, y_pred) == mean_squared_error(y_true, y_pred)


def test_categorical_cross_entropy():
    y_true, y_pred, _ = get_rand_iris_predictions()
    loss = CategoricalCrossEntropy()

    assert np.isclose(
        loss.evaluate(y_true, y_pred),
        log_loss(y_true.argmax(axis=1), y_pred, labels=np.arange(y_pred.shape[1])),
    )


test_categorical_cross_entropy()
