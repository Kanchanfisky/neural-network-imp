import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self):
        self.forwarded_values = {}
        self.parameters = {}
        self.gradients = {}

    def initialize_parameters(self):
        biases = []
        weights = [np.array([[4, -10, 3], [3, 6, 1]]), np.array([[-1, 8]])]

        biases.append(np.array([[10], [5]]))
        biases.append(np.array([[-10]]))

        # return weights, biases

        self.parameters = {
            "w1": weights[0],
            "w2": weights[1],
            "b1": biases[0],
            "b2": biases[1]
        }

        return self.parameters

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + (np.exp(-x)))

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def derivative_sigmoid(x):
        return NeuralNetwork.sigmoid(x) * (1.0 - NeuralNetwork.sigmoid(x))

    @staticmethod
    def derivative_relu(x):
        return np.array(x > 0, dtype=np.float32)

    def forward_propagation(self, x):
        w1 = self.parameters['w1']
        b1 = self.parameters['b1']
        w2 = self.parameters['w2']
        b2 = self.parameters['b2']

        z1 = np.dot(w1, x) + b1
        a1 = NeuralNetwork.relu(z1)

        z2 = np.dot(w2, a1) + b2
        a2 = NeuralNetwork.sigmoid(z2)

        self.forwarded_values = {
            "z1": z1,
            "a1": a1,
            "z2": z2,
            "a2": a2
        }

        return self.forwarded_values

    def cost_function(self, a2, y):
        # no of columns in y
        size = y.shape[0]
        cost = (1.0 / size) * np.sum((y - a2) ** 2)
        return cost

    # def cost_function_with_log(self, a2, y):
    #     m = y.shape[1]
    #
    #     cost = -(1 / m) * np.sum(y * np.log(a2))
    #
    #     return cost

    def backward_propagation(self, x, y):
        w1 = self.parameters['w1']
        b1 = self.parameters['b1']
        w2 = self.parameters['w2']
        b2 = self.parameters['b2']

        a1 = self.forwarded_values['a1']
        a2 = self.forwarded_values['a2']
        z1 = self.forwarded_values['z1']
        z2 = self.forwarded_values['z2']

        size = x.shape[1]

        # delta = self.cost_derivative(activations[-1], y) * \
        #     sigmoid_prime(zs[-1])
        # nabla_b[-1] = delta
        # nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        dy_hat = (a2 - y)
        delta = dy_hat * NeuralNetwork.derivative_sigmoid(z2)  # dC by da2 then da2 by dz2

        db2 = np.sum(delta) / (float(size))
        dw2 = np.dot(delta, a1.transpose()) * (1 / float(size))
        # dw2 = np.matmul(a1, delta.T) * (1/float(size))

        dw1 = np.dot(delta * NeuralNetwork.derivative_sigmoid(z1), x.T) * (1 / float(size))
        db1 = np.sum(delta * NeuralNetwork.derivative_sigmoid(z1)) * 1 / (float(size))

        gradients = {
            "dw1": dw1,
            "db1": db1,
            "dw2": dw2,
            "db2": db2
        }

        return gradients

    def update_parameters(self, learning_rate):
        w1 = self.parameters['w1']
        b1 = self.parameters['b1']
        w2 = self.parameters['w2']
        b2 = self.parameters['b2']

        dw1 = self.gradients['dw1']
        db1 = self.gradients['db1']
        dw2 = self.gradients['dw2']
        db2 = self.gradients['db2']

        w1 = w1 - learning_rate * dw1
        b1 = b1 - learning_rate * db1
        w2 = w2 - learning_rate * dw2
        b2 = b2 - learning_rate * db2

        parameters = {
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2
        }

        return parameters

    def show_cost_function_plot(self, cost, iterations):
        t = np.arange(0, iterations)
        plt.plot(t, cost)
        plt.xlabel("iterations")
        plt.ylabel("Cost")
        plt.show()

    def train(self, x, y, learning_rate, iterations):
        n_x = x.shape[0]
        n_y = y.shape[0]

        cost = []

        parameters = self.initialize_parameters()
        for i in range(iterations):
            self.forwarded_values = self.forward_propagation(x)
            cost_value = self.cost_function(self.forwarded_values['a2'], y)
            self.gradients = self.backward_propagation(x, y)
            self.parameters = self.update_parameters(learning_rate)

            cost.append(cost_value)

            # if i % (iterations / 10) == 0:
            #     print("Cost after", i, "iterations is :", cost_value)
            print("Cost after", i, "iterations is :", cost_value)
        return parameters, cost
