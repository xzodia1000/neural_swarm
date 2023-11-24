import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import LabelBinarizer
from neural_swarm.ann.activation import Identity, Relu, Sigmoid, Softmax, Tanh
from neural_swarm.ann.ann import ANN
from neural_swarm.ann.loss import (
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    Hinge,
    Mse,
)
from neural_swarm.constants import (
    LOCAL_GLOBAL_INFORMANTS,
    LOCAL_NEIGHBOUR_INFORMANTS,
    RANDOM_INFORMANTS,
)
from neural_swarm.pso.ann_function import ANNFunction
from neural_swarm.pso.pso import PSO


class ANNMain:
    def __init__(self):
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.ann = None
        self.first_layer = None
        self.last_layer = None
        self.last_layer_activation = None
        self.loss = None

        self.pso = None
        self.ann_function = None

    def set_dataset(self, dataset):
        self.dataset = dataset

    def get_dataset_features(self):
        features = self.dataset.columns[:-1]
        return features.tolist()

    def set_inital_values(self, test_size=0.2):
        df = self.dataset.sample(frac=1)
        X = df.drop(df.columns[-1], axis=1).values
        y = df[df.columns[-1]].values

        self.set_layers_and_loss(X, y, test_size)

    def set_layers_and_loss(self, X, y, test_size=0.2):
        if y.dtype == "float":
            self.loss = Mse()
            self.last_layer_activation = Identity()
            self.last_layer = 1
            self.first_layer = X.shape[1]
        else:
            le = LabelEncoder()
            lb = LabelBinarizer()
            encoded_y = le.fit_transform(y)
            classes = np.unique(encoded_y)

            if len(classes) == 2:
                y = y.astype(int)
                self.loss = BinaryCrossEntropy()
                self.last_layer_activation = Sigmoid()
                self.last_layer = 1
                self.first_layer = X.shape[1]
            elif len(classes) > 2:
                y = lb.fit_transform(y)
                self.loss = CategoricalCrossEntropy()
                self.last_layer_activation = Softmax()
                self.last_layer = len(classes)
                self.first_layer = X.shape[1]

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

    def update_loss(self, loss):
        if self.loss.__str__() == "BinaryCrossEntropy":
            if loss == "Hinge":
                self.loss = Hinge()

    def set_ann(self, layers):
        activation_options = [Sigmoid(), Tanh(), Relu(), Softmax(), Identity()]
        self.ann = ANN()
        for layer in layers:
            self.ann.add_layer(activation_options[layer["activation"]], layer["nodes"])

    def get_layers(self):
        layers = []

        for layer in self.ann.network.layers:
            layers.append(
                {"nodes": layer.nodes, "activation": layer.activation.__str__()}
            )

        return layers

    def set_pso(
        self,
        randomize_weights,
        swarm_size,
        informants_type,
        informants_size,
        iterations,
        optimize_activation,
        opt,
        alpha,
        beta,
        gamma,
        delta,
        epsilon,
    ):
        self.ann_function = ANNFunction(
            self.ann, self.X_train, self.y_train, self.loss, optimize_activation
        )

        informants_type_map = {
            "Random": RANDOM_INFORMANTS,
            "Neighbours": LOCAL_NEIGHBOUR_INFORMANTS,
            "Global Best and Neighbours": LOCAL_GLOBAL_INFORMANTS,
        }

        opt = "MAX" if opt == "Maximize" else "MIN"

        if randomize_weights:
            alpha = np.random.uniform(0, 1)
            beta, gamma, delta = np.random.dirichlet(np.ones(3)) * 4
            epsilon = np.random.uniform(0.4, 0.9)

        self.pso = PSO(
            self.ann_function,
            swarm_size,
            alpha,
            beta,
            gamma,
            delta,
            epsilon,
            informants_type_map[informants_type],
            informants_size,
            iterations,
            opt,
        )

    def train_ann(self):
        loss = []
        acc = []
        i = 0
        for l, a, p in self.pso.evolve():
            loss.append(l)
            acc.append(a)
            particles = [i.position for i in p]
            i += 1
            yield i, loss, acc, particles
