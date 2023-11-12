import numpy as np
from neural_swarm.ann.loss import BinaryCrossEntropy
from test.test_data import prep_data, prep_test_ann


def test_ann():
    loss_function = BinaryCrossEntropy()
    ann = prep_test_ann()
    x, y_true = prep_data()

    y_pred = ann.run(x)
    print("Predicted values: ", y_pred)

    loss, acc = ann.evaluate(x, y_true, loss_function)
    print("Loss: ", loss)
    print("Accuracy: ", acc)
