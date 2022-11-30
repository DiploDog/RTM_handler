import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse


class PolyRegression:

    def __init__(self, data, x, y, f_name, appendix, degree=2):
        self.data = data
        self.x = x
        self.y = y
        self.f_name = f_name
        self.degree = degree
        self.appendix = appendix
        self.coeffs = None

        self.error = None
        self.y_pred = None

    def generate_degrees(self, source_data, degree):
        return np.array([
            source_data**n for n in range(1, degree + 1)
        ]).T

    def train_polynomial(self):
        try:
            X = self.generate_degrees(self.x, self.degree)
        except KeyError:
            print('No key for x in existing data!')
        try:
            model = LinearRegression().fit(X, self.y)
            self.coeffs = [model.coef_, model.intercept_]
        except KeyError:
            print('No key for y in existing data!')
        else:
            self.y_pred = model.predict(X)
            self.error = mse(self.y, self.y_pred)

    def plot_graph(self, xlabel, ylabel):
        try:
            if self.error is None:
                raise ValueError
        except ValueError:
            print('Loss value is None')
        else:
            title = 'Степень полинома %d, Среднеквадратическая ошибка %.3f' \
                    % (self.degree, self.error)

            plt.scatter(self.x, self.y, 5, 'g', 'o', alpha=0.8, label='data')
            plt.plot(self.x, self.y_pred)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            #plt.ylim([0, -1])
            plt.title(title)
            plt.savefig('temp/curves/' + self.f_name.split('.')[0] + '_' + self.appendix + '.jpg')
            plt.clf()

    def get_coeffs_(self):
        return self.coeffs

