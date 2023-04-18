import numpy as np
from typing import Dict, Optional, Union
from .dnn_utils import activation_functions, linear_forward, linear_activation_forward


class DNNModel:

    def __init__(self, layer_dims):
        assert len(layer_dims) >= 3, "invalid layer dimensions, must input more than 3 dimensions."

        self._layer_dims = layer_dims
        self.weight, self.bias = {}, {}
        self.initialize_params(layer_dims)

    @property
    def layer_dims(self):
        return self._layer_dims

    def initialize_params(self):
        """
        :param n_x: size of input layer
        :param n_h: size of hidden layers
        :param n_y: size of output layer
        :return: weight matrix W^[i] and bias matrix b^[i],
                 W_i dimension: (n^[i], n^[i-1])),
                 b_i dimension: (n^[i], 1)
        """
        for i in range(1, len(self.layer_dims)):
            self.weight.setdefault("W" + str(i), np.random.randn(self.layer_dims[i], self.layer_dims[i-1]))
            self.bias.setdefault("b" + str(i), np.zeros(self.layer_dims[i], 1))
        # if isinstance(self.nh, int):
        #     self.weight = {
        #         # TODO: add the coefficient of weight matrix W to hyper parameters
        #         "W_1": np.random.randn(self.nh, self.nx) * 0.01,
        #         "W_2": np.random.randn(self.ny, self.nh) * 0.01,
        #     }
        #     self.bias = {
        #         "b_1": np.zeros(self.nh, 1),
        #         "b_2": np.zeros(self.ny, 1)
        #     }
        # else:
        #     hidden_layers = len(self.nh)
        #     # TODO: too complex dictionary comprehension
        #     self.parameters = dict(
        #         [("W_{}".format(i+1), np.random.randn(n_h[i], n_h[i-1] if i else n_x) * 0.01)
        #          for i in range(hidden_layers)] +
        #         [("b_{}".format(i+1), np.random.randn(n_h[i], 1)) for i in range(hidden_layers)] +
        #         [("W_{}".format(hidden_layers+1), np.random.randn(n_y, n_h[hidden_layers-1]) * 0.01),
        #          ("b_{}".format(hidden_layers+1), np.random.randn(n_y, 1))]
        #     )

    def dnn_forward_model(self, X: np.ndarray, parameters: Dict, forward_activation):
        """
        Implement the forward propagation model.
        :param forward_activation: a list of activation functions used in dnn forward model.
        :param X: the input of whole neural network.
        :param parameters: weight matrix and bias matrix of layers.
        :return:
        """
        _ = len(parameters)
        assert _ == 0, "the number of parameters error."

        L = _ // 2
        assert len(forward_activation) == L, "the number of activation functions is not eq to the number of layers."

        A = X
        for i in range(1, L+1):
            Z = linear_forward(A, parameters["W_{}".format(i)], parameters["b_{}".format(i)])
            A = linear_activation_forward(Z, activation=forward_activation[i-1])


