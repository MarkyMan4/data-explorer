import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

class LinRegressor:

    dataset = []
    lin_reg = None

    def __init__(self, dataset):
        self.dataset = dataset
        self.ir = LinearRegression()

    def select_columns(self, x, y):
        self.dataset = self.dataset.sort_values(x)
        x = np.array(self.dataset[[x]])
        y = np.array(self.dataset[y])

        return x, y

    def get_graph(self, col1, col2):
        x, y = self.select_columns(col1, col2)

        self.lin_reg = LinearRegression()
        self.lin_reg.fit(x, y)
        pred = self.lin_reg.predict(x)

        fig = plt.figure(figsize=(15,8))
        raw = plt.scatter(x, y, color='red')
        fit, = plt.plot(x, pred, linewidth=3, color='blue')
        plt.legend((raw, fit), ('Data', 'Linear Fit'), loc='lower right')
        plt.title('Linear regression', fontsize=24)
        plt.xlabel(col1, fontsize=18)
        plt.ylabel(col1, fontsize=18)
        
        return fig

    def make_prediction(self, input):
        feature = [[input]]
        pred = self.lin_reg.predict(feature)

        return pred
