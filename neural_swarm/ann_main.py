import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import LabelBinarizer
from neural_swarm.ann.activation import Identity, Relu, Sigmoid, Softmax, Tanh
from neural_swarm.ann.ann import ANN
from neural_swarm.ann.loss import (
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
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
        """Initializes the main class for ANN model training using PSO."""
        # Data and split sets
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # ANN and layers setup
        self.ann = None
        self.first_layer = None
        self.last_layer = None
        self.last_layer_activation = None
        self.loss = None

        # PSO setup
        self.pso = None
        self.ann_function = None
        self.iterations = None
        self.ann_loss = []  # To store loss history
        self.ann_acc = []  # To store accuracy history

    def set_dataset(self, dataset):
        """Sets the dataset for training the ANN.

        Args:
            dataset: The dataset to be used for training.
        """
        self.dataset = dataset

    def get_dataset_features(self):
        """Returns the feature names of the dataset.

        Returns:
            List: A list of feature names.
        """
        features = self.dataset.columns[:-1]
        return features.tolist()

    def set_inital_values(self, test_size=0.2):
        """Initializes training and test sets and sets up layers and loss function.

        Args:
            test_size (float, optional): Proportion of the dataset to include in the test split.
        """
        df = self.dataset.sample(frac=1)  # Shuffle the dataset
        X = df.drop(df.columns[-1], axis=1).values  # Input features
        y = df[df.columns[-1]].values  # Output labels

        self.set_layers_and_loss(X, y, test_size)  # Setup layers and loss function

    def set_layers_and_loss(self, X, y, test_size=0.2):
        """Determines the appropriate loss function and last layer activation based on the label type.

        Args:
            X: Input features.
            y: Output labels.
            test_size (float, optional): Proportion of the dataset to include in the test split.
        """
        # Determine loss function and layer setup based on label data type
        if y.dtype == "float":
            # Regression setup
            self.loss = Mse()
            self.last_layer_activation = Identity()
            self.last_layer = 1
            self.first_layer = X.shape[1]
        else:
            # Classification setup
            le = LabelEncoder()
            lb = LabelBinarizer()
            encoded_y = le.fit_transform(y)
            classes = np.unique(encoded_y)

            # Binary classification
            if len(classes) == 2:
                y = y.astype(int)
                self.loss = BinaryCrossEntropy()
                self.last_layer_activation = Sigmoid()
                self.last_layer = 1
                self.first_layer = X.shape[1]
            # Multi-class classification
            elif len(classes) > 2:
                y = lb.fit_transform(y)
                self.loss = CategoricalCrossEntropy()
                self.last_layer_activation = Softmax()
                self.last_layer = len(classes)
                self.first_layer = X.shape[1]

            # Split the dataset into training and testing sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

    def set_ann(self, layers):
        """Sets up the ANN with the specified layers.

        Args:
            layers: A list of dictionaries specifying the layers' configurations.
        """
        activation_options = [Sigmoid(), Tanh(), Relu(), Softmax(), Identity()]
        self.ann = ANN()
        # Add each layer to the ANN
        for layer in layers:
            self.ann.add_layer(activation_options[layer["activation"]], layer["nodes"])

    def get_layers(self):
        """Returns the configuration of layers in the ANN.

        Returns:
            List: A list of dictionaries containing layer configurations.
        """
        layers = []
        # Collect layer configurations
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
        """Initializes the PSO algorithm with the specified parameters.

        Args:
            randomize_weights: Flag to randomize weights.
            swarm_size: Number of particles in the swarm.
            informants_type: Type of informant selection strategy for each particle.
            informants_size: Number of informants for each particle.
            iterations: Number of iterations to run the optimization.
            optimize_activation: Flag to optimize activation functions.
            opt: Optimization operator (minimize or maximize).
            alpha, beta, gamma, delta, epsilon: Coefficients for the velocity and position update equations.
        """
        self.ann_function = ANNFunction(
            self.ann, self.X_train, self.y_train, self.loss, optimize_activation
        )

        # Map for converting informant type to corresponding constant
        informants_type_map = {
            "Random": RANDOM_INFORMANTS,
            "Neighbours": LOCAL_NEIGHBOUR_INFORMANTS,
            "Global Best and Neighbours": LOCAL_GLOBAL_INFORMANTS,
        }

        opt = "MAX" if opt == "Maximize" else "MIN"  # Optimization direction

        self.iterations = iterations  # Number of iterations

        # Randomize weights if required
        if randomize_weights:
            alpha = np.random.uniform(0, 1)
            beta, gamma, delta = np.random.dirichlet(np.ones(3)) * 4
            epsilon = np.random.uniform(0.4, 0.9)

        # Initialize PSO with the specified parameters
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
        """Trains the ANN using PSO.

        Yields:
            int: The iteration number for each training step.
        """
        i = 0
        # Iterate through PSO evolution
        for l, a in self.pso.evolve():
            self.ann_loss.append(l)  # Record loss
            self.ann_acc.append(a)  # Record accuracy
            i += 1
            yield i  # Yield iteration number

    def evaluate_results(self):
        """Evaluates the trained ANN on the test set.

        Returns:
            Tuple: The accuracy and loss on the test set.
        """
        return self.ann.evaluate(
            self.X_test, self.y_test, self.loss
        )  # Evaluate the ANN
