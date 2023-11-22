import numpy as np
from math import exp
import pandas as pd

class Neuron_Layer():
    def __init__(self, num_neurons: int, weights_per_neuron: int):
        self.input_values = np.array([0.0]*num_neurons)
        self.act_values = np.array([0.0]*num_neurons)
        self.deltas = np.array([0.0]*num_neurons)
        self.weight_matrix = np.random.rand(num_neurons, weights_per_neuron) * 0.1-0.05
        self.neuron_count = num_neurons

    """
    Activation function
    """
    def __sigmoid(self, x: float) -> float:
        return 1/(1 + exp(-x))
    
    """
    Derivative of the activation function
    """
    def __sigmoid_prime(self, x: float) -> float:
        return self.__sigmoid(x)*(1 - self.__sigmoid(x))

    """
    Calculates all input and activation values for the layer
    recieves as an argument the activation vector of the 
    previous layer
    """
    def activate_layer(self, prev_activation: np.ndarray):
        if len(prev_activation) != len(self.weight_matrix[0]):
            raise Exception("Weight vector must be same length of previous layer activation")
        
        for i in range(len(self.weight_matrix)):
            #Recieve neuron input
            self.input_values[i] = np.dot(self.weight_matrix[i], prev_activation)
            #Calculate activation value
            self.act_values[i] = self.__sigmoid(self.input_values[i])

    """
    Updates delta vector for the output layer. Do not use with
    hidden layers.
    """
    def update_delta_output(self, err_vec: np.ndarray) -> None:
        for i in range(self.neuron_count):
            self.deltas[i] = (err_vec[i] - self.act_values[i])*self.__sigmoid_prime(self.input_values[i])
        return
    
    """
    Update the delta for a hidden layer. Recieves the next layer in the
    network in order to access its delta and weights
    """
    def update_delta_hidden(self, next_layer: 'Neuron_Layer') -> None:
        for i in range(self.get_neuron_count()):
            sum = 0.0
            for j in range(next_layer.get_neuron_count()):
                sum += (next_layer.weight_matrix[j][i])*(next_layer.get_delta()[j])
            self.deltas[i] = self.__sigmoid_prime(self.input_values[i])*sum
    """
    Updates the weight matrix for the network layer
    """
    def update_weights(self, prev_activations: np.ndarray, learning_rate: float) -> None:
        #Update each element in the weight matrix
        for i in range(len(self.weight_matrix)):
            for j in range(len(self.weight_matrix[i])):
                self.weight_matrix[i][j] += learning_rate*prev_activations[j]*self.deltas[i]
    
    """
    Returns the input vector of the layer
    """
    def get_inputs(self) -> np.ndarray:
        return self.input_values
    
    """
    Returns the activation vector of the layer
    """
    def get_activations(self) -> np.ndarray:
        return self.act_values
    
    """
    Returns the local gradient of the layer
    """
    def get_delta(self) -> np.ndarray:
        return self.deltas
    
    """
    Return the number of neurons in the layer
    """
    def get_neuron_count(self) -> int:
        return self.neuron_count