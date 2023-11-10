from Perceptron import Perceptron
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class Neural_Network():
    """
        Implements a Neural Network.

        The neural network is represented by a ndarray. Each row
        of the array represents a layer of the network.

        Output layer is the last row.

        The constructor recieve the number of hidden layers (default 0) and 
        a list of number of perceptrons per layer (default 1). 
    """
    def __init__(self, 
                 examples: pd.DataFrame,
                 dependent_var: str,
                 indepedent_vars: [str],
                 neurons_per_layer: [int],
                 num_hidden_layers: int = 0, 
                 ) -> None:
        if num_hidden_layers != len(neurons_per_layer) - 1:
            raise Exception("Error: Number of hidden layer must be equal to the lenght of the neurons per layer minus one")
        
        self.examples = examples
        self.dependent_var = dependent_var
        self.independent_vars = indepedent_vars
        self.neurons_per_layer = neurons_per_layer
        self.num_hidden_layers = num_hidden_layers

        num_indepedent_vars = len(indepedent_vars)
        
        for i in range(0,len(neurons_per_layer)):
            if i == 0:
                self.network = [[Perceptron(num_indepedent_vars)]*neurons_per_layer[i]]
            else:
                num_prev_layer = len(self.network[i-1])
                self.network.append([Perceptron(num_prev_layer)]*neurons_per_layer[i])

    def cross_validation(self, size_of_training_set: int) -> None:
        x = self.examples[self.independent_vars]
        y = self.examples[self.dependent_var]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size_of_training_set)

        # training
        self.__backpropagation(x_train, y_train)

        # testing


    def __backpropagation(self, training_set: pd.DataFrame, training_answers: pd.DataFrame) -> None:
        return None
    
    def __evaluate_network(self, input_values: np.array) -> None:
        a = input_values


