import numpy as np
import pandas as pd
from sklearn.naive_bayes import LabelBinarizer
from neural_swarm.ann.activation import Sigmoid, Softmax
from neural_swarm.ann.ann import ANN


def prep_data():
    df = pd.read_csv("input/data_banknote_authentication.txt", header=None)
    df = df.sample(frac=1)
    x = df.drop(df.columns[-1], axis=1).values
    y_true = df[df.columns[-1]].values

    return x, y_true


def get_rand_predictions():
    _, y_true = prep_data()
    y_pred = np.random.rand(len(y_true))

    return y_true, y_pred


def get_randn_predictions():
    _, y_true = prep_data()
    y_pred = np.random.randn(len(y_true))

    return y_true, y_pred


def prep_test_iris():
    label_binarizer = LabelBinarizer()
    df = pd.read_csv("input/iris.data.csv", header=None)
    df = df.sample(frac=1)
    x = df.drop(df.columns[-1], axis=1).values
    y_true = label_binarizer.fit_transform(df[df.columns[-1]].values)

    return x, y_true, label_binarizer.classes_


def get_rand_iris_predictions():
    _, y_true, labels = prep_test_iris()
    y_pred = y_true.copy()

    return y_true, y_pred, labels


def prep_test_ann():
    activation = Sigmoid()

    ann = ANN()
    ann.add_layer(nodes=4, activation=activation)
    ann.add_layer(nodes=2, activation=activation)
    ann.add_layer(nodes=3, activation=activation)
    ann.add_layer(nodes=1, activation=activation)

    return ann


def prep_test_iris_ann():
    activation = Softmax()
    activation_sigmoid = Sigmoid()

    ann = ANN()
    ann.add_layer(nodes=4, activation=activation_sigmoid)
    ann.add_layer(nodes=3, activation=activation_sigmoid)
    ann.add_layer(nodes=2, activation=activation_sigmoid)
    ann.add_layer(nodes=3, activation=activation)

    return ann
