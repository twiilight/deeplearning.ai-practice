import numpy as np
from typing import Dict, Optional, Union
from .dnn_utils import activation_functions, linear_forward, linear_activation_forward


class DNNModel:

    def __init__(self, layer_dims):
        assert len(layer_dims) >= 3, "invalid layer dimensions, must input more than 3 dimensions."

        self._layer_dims = layer_dims
        self.weight, self.bias = {}, {}
        self.initialize_params()

    @property
    def layer_dims(self):
        return self._layer_dims

    @property
    def layers(self):
        return len(self.layer_dims)

    def initialize_params(self):
        """
        :param n_x: size of input layer
        :param n_h: size of hidden layers
        :param n_y: size of output layer
        :return: weight matrix W^[i] and bias matrix b^[i],
                 W_i dimension: (n^[i], n^[i-1])),
                 b_i dimension: (n^[i], 1)
        """
        for i in range(1, self.layers):
            self.weight.setdefault("W" + str(i), np.random.randn(self.layer_dims[i], self.layer_dims[i-1]))
            self.bias.setdefault("b" + str(i), np.zeros(self.layer_dims[i], 1))

        assert len(self.weight) % 2 and len(self.bias) % 2, "the number of parameters error."

    def dnn_forward_model(self, X: np.ndarray, forward_activation=None):
        """
        Implement the forward propagation model.
        :param forward_activation: a list of activation functions used in dnn forward model.
        :param X: the input of whole neural network.
        :return:
        """
        if not forward_activation:
            forward_activation = ["relu" for i in range(self.layers - 1)]

        assert len(forward_activation) == self.layers - 1, \
            "the number of activation functions is not eq to the number of layers."

        A = X
        for i in range(1, self.layers):
            Z = linear_forward(A, self.weight["W{}".format(i)], self.bias["b{}".format(i)])
            A = linear_activation_forward(Z, activation=forward_activation[i-1])


