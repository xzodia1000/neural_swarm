from neural_swarm.ann.loss import BinaryCrossEntropy, CategoricalCrossEntropy
from test.test_data import prep_data, prep_test_ann, prep_test_iris, prep_test_iris_ann


def test_ann():
    loss_function = BinaryCrossEntropy()
    ann = prep_test_ann()
    x, y_true = prep_data()

    y_pred = ann.run(x)
    print("Predicted values: ", y_pred)

    acc, loss = ann.evaluate(x, y_true, loss_function)
    print("Loss: ", loss)
    print("Accuracy: ", acc)


def test_iris_ann():
    loss_function = CategoricalCrossEntropy()
    ann = prep_test_iris_ann()
    x, y_true, _ = prep_test_iris()

    y_pred = ann.run(x)
    print("Predicted values: ", y_pred)

    acc, loss = ann.evaluate(x, y_true, loss_function)
    print("Loss: ", loss)
    print("Accuracy: ", acc)
