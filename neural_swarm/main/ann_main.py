import pandas as pd
from sklearn.model_selection import train_test_split
from neural_swarm.ann.ann import ANN


class ANNMain:
    def __init__(self):
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.ann = ANN()
        self.loss = None

    def set_inital_values(self, dataset, header=False, test_size=0.2):
        df = pd.read_csv(dataset, header=None if header else 0)
        df = df.sample(frac=1)
        X = df.drop(df.columns[-1], axis=1).values
        y = df[df.columns[-1]].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
    def set_loss(self, y):
        pass
            
