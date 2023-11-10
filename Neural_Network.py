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
        self.num_layers = num_hidden_layers+1

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

        x_train['constant'] = 1
        x_test['constant'] = 1

        training_data = pd.concat([x_train, y_train], axis=1)
        testing_data = pd.concat([x_test, y_test], axis=1)
        # training
        self.__backpropagation(training_data)

        # testing


    """
        Implementation of backpropagaion algorithm to train a neural network
    """
    def __backpropagation(self, training_set: pd.DataFrame) -> None:
        i = 0
        while (True):
            for (index,row) in training_set.iterrows():
                x = row[self.independent_vars+['constant']]
                y = row[self.dependent_var]

                # activation of each perceptron
                a = [np.array(x)] # activation for input layer
                prev_a = a[0]
                for i in range(0, self.num_layers):
                    current_layer = self.network[i]
                    current_a = [p.input_function(prev_a) for p in current_layer] + [1]
                    prev_a = current_a
                    a.append([current_a])

                ## propagar el descenso del gradiente
                ## todo

            if i == 100000:
                break

            i += 1
    
    def __evaluate_network(self, input_values: np.array) -> None:
        return None


