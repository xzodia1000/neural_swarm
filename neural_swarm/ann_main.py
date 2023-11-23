import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from ann.activation import Identity, Sigmoid, Softmax
from ann.ann import ANN
from ann.loss import BinaryCrossEntropy, CategoricalCrossEntropy, Mse


class ANNMain:
    def __init__(self):
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.ann = ANN()
        self.first_layer = None
        self.last_layer = None
        self.last_layer_activation = None
        self.loss = None

    def set_dataset(self, dataset):
        self.dataset = dataset

    def get_dataset_features(self):
        features = self.dataset.columns[:-1]
        return features.tolist()

    def set_inital_values(self, test_size=0.2):
        df = self.dataset.sample(frac=1)
        X = df.drop(df.columns[-1], axis=1).values
        y = df[df.columns[-1]].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        self.set_layers_and_loss(X, y)

    def set_layers_and_loss(self, X, y):
        if y.dtype == "float":
            self.loss = Mse()
            self.last_layer_activation = Identity()
            self.last_layer = 1
            self.first_layer = X.shape[1]
        else:
            le = LabelEncoder()
            encoded_y = le.fit_transform(y)
            classes = np.unique(encoded_y)

            if len(classes) == 2:
                self.loss = BinaryCrossEntropy()
                self.last_layer_activation = Sigmoid()
                self.last_layer = 1
                self.first_layer = X.shape[1]
            elif len(classes) > 2:
                self.loss = CategoricalCrossEntropy()
                self.last_layer_activation = Softmax()
                self.last_layer = len(classes)
                self.first_layer = X.shape[1]

    def get_column_names(self):
        return self.dataset.columns
