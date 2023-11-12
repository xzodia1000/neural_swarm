from sklearn.metrics import hinge_loss, log_loss, mean_squared_error
from neural_swarm.ann.loss import BinaryCrossEntropy, Mse, Hinge
from test.test_data import get_rand_predictions, get_randn_predictions


def test_binary_cross_entropy():
    y_true, y_pred = get_rand_predictions()
    loss = BinaryCrossEntropy()
    assert loss.evaluate(y_true, y_pred) == log_loss(y_true, y_pred)


def test_mse():
    y_true, y_pred = get_randn_predictions()
    loss = Mse()
    assert loss.evaluate(y_true, y_pred) == mean_squared_error(y_true, y_pred)


def test_hinge():
    y_true, y_pred = get_randn_predictions()
    loss = Hinge()
    assert loss.evaluate(y_true, y_pred) == hinge_loss(y_true, y_pred)
