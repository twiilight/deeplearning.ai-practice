import numpy as np


def sigmoid(Z: np.ndarray) -> np.ndarray:
    """
    Implements the activation function of sigmoid in numpy.
    :param Z: the input array of activation function, the dimension of Z is (n_i, m)
    :return: A, the output array of activation function, the dimension of A is (n_i, m)
    """
    return 1 / (1 + np.exp(-Z))


def relu(Z: np.ndarray) -> np.ndarray:
    """
    Implements the activation function of ReLU in numpy.
    :param Z: the input array of activation function, the dimension of Z is (n_i, m)
    :return: A, the output array of activation function, the dimension of A is (n_i, m)
    """
    return np.max(0, Z)


activation_functions = {
    "sigmoid": sigmoid,
    "relu": relu
}


def linear_forward(A: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Implement the linear part of a layer's forward propagation.
    :param A: activation from the previous layer. (n^[i-1], m)
    :param W: weight matrix of current layer. (n^[i], n^[i-1])
    :param b: bias matrix of current layer. (n^[i], 1)
    :return: Z = W.*X + b (n_[i], m)
    """
    Z = np.dot(W, A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))

    return Z


def linear_activation_forward(Z: np.ndarray, activation):
    """
    Implement the forward propagation of linear to activation layer.
    :param Z: the linear output of a layer. (n^[i], m)
    :param activation: the form of activation function.
    :return: A = g^[i](Z)
    """
    if activation in activation_functions:
        return activation_functions[activation](Z)
    else:
        raise KeyError("No activation function named " + activation)
