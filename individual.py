import numpy as np


def sigmoid_activate(layer):
    return 1 / (1 + np.exp(-layer))


def softmax_activate(layer):
    m = np.exp(layer)
    return m / m.sum(len(layer.shape) - 1)


class Individual:
    def __init__(self, input_size, hidden_layers_num, hidden_layers_neurons, is_copy=False):
        self.score = 0
        self.time_alive = 0
        self.move_frequency = {0: 0, 1: 0, 2: 0}
        self.bonus_fitness = 0
        self.layers = []
        self.biases = []

        if not is_copy:
            for i in range(hidden_layers_num ):
                entry_size = hidden_layers_neurons if i != 0 else input_size
                self.layers.append(np.random.rand(hidden_layers_neurons, entry_size) * 2 - 1)
                self.biases.append(np.random.rand(hidden_layers_neurons, 1) * 2 - 1)

            self.outputs = np.random.rand(3, hidden_layers_neurons) * 2 - 1

    def forward(self, inputs):
        inputs = inputs.astype(float).reshape((-1, 1))

        for layer, bias in zip(self.layers, self.biases):
            inputs = np.matmul(layer, inputs)
            inputs = inputs + bias
            inputs = sigmoid_activate(inputs)

        inputs = np.matmul(self.outputs, inputs)
        inputs = inputs.reshape(-1)
        return softmax_activate(inputs)
