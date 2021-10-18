import numpy as np

from NeuralNetwork import NeuralNetwork


# RUN this class to run the solution

if __name__ == '__main__':
    x_train = np.array([[1, 2, 0],
                        [2, 5, 0],
                        [0, 0, 3],
                        [4, 3, 0],
                        [0, 0, 5],
                        [0, 0, 1]])

    x_trans = np.transpose(x_train)
    # print(x_trans)

    y_train = np.array([1, 1, 0, 1, 0, 0])
    # print(y_train.shape)
    # layers = [3, 2, 1]

    nn = NeuralNetwork()
    iters = 1000
    learning_rate = 1
    params, cost = nn.train(x_trans, y_train, learning_rate, iters)

    nn.show_cost_function_plot(cost, iters)

    # parameters = nn.initialize_parameters()
    # # print(parameters)
    #
    # forward_params = nn.forward_propagation(x_trans)
    # print(forward_params)
    #
    # # #print('cost function value = ' + str(nn.cost_function(forward_params["a2"], y_train)))
    # # print(nn.cost_function(forward_params["a2"], y_train))
    # print(nn.backward_propagation(x_trans, y_train))
