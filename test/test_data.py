import numpy as np
import pandas as pd


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
