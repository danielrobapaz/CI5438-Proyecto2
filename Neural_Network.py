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
        if num_hidden_layers != len(neurons_per_layer):
            raise Exception("Error: Number of hidden layer must be equal to the lenght of the neurons per layer minus one")
        
        self.examples = examples
        self.dependent_var = dependent_var
        self.independent_vars = indepedent_vars
        self.neurons_per_layer = neurons_per_layer
        self.num_layers = num_hidden_layers+1
        self.network = []
        num_indepedent_vars = len(indepedent_vars)
        
        for i in range(0,len(neurons_per_layer)):
            print(i)
            if i == 0:
                current_layer = [Perceptron(num_indepedent_vars) for j in range(neurons_per_layer[i])]
            else:
                num_prev_layer = len(self.network[i-1])
                current_layer = [Perceptron(num_prev_layer) for j in range(neurons_per_layer[i])]
            self.network.append(current_layer)

        self.network.append([Perceptron(neurons_per_layer[num_hidden_layers-1])])

    def cross_validation(self, size_of_training_set: float,
                         learning_rate: float = 0.1) -> None:
        x = self.examples[self.independent_vars]
        y = self.examples[self.dependent_var]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size_of_training_set)

        x_train['constant'] = 1
        x_test['constant'] = 1

        training_data = pd.concat([x_train, y_train], axis=1)
        testing_data = pd.concat([x_test, y_test], axis=1)
        # training
        self.__backpropagation(training_data, learning_rate)

        # testing


    """
        Implementation of backpropagaion algorithm to train a neural network
    """
    def __backpropagation(self, training_set: pd.DataFrame,
                          learning_rate: float) -> None:
        epoch = 0
        while (True):
            for (index,row) in training_set.iterrows():
                x = row[self.independent_vars+['constant']]
                y = row[self.dependent_var]
                # activation of each perceptron
                activation_per_perceptron = [list(x)] # activation for input layer
                input_per_perceptron = []
                for i in range(0, self.num_layers):
                    current_layer = self.network[i]
                    previous_activation = activation_per_perceptron[-1]

                    # input for each neuron
                    current_input = [p.input_function(previous_activation) for p in current_layer]
                    
                    # activation of each neuron
                    current_activation = [p.activation_function(previous_activation) for p in current_layer] + [1]
                    
                    activation_per_perceptron.append(current_activation)
                    input_per_perceptron.append(current_input)

                # propagates delta from output layer
                delta = []
                for j in range(len(self.network[-1])):
                    current_perceptron = self.network[-1][j]
                    current_activation = activation_per_perceptron[-1][j]
                    current_input = input_per_perceptron[-1][j]
                    derivate = current_perceptron.activation_function_derivate(current_input)
                    error = y - current_activation
                    delta.append(derivate*error)
                
                delta = [delta]
                current_delta_index = 0
                for l in range(self.num_layers-2, -1, -1):
                    current_layer = self.network[l]
                    current_gradient = delta[0]
                    current_delta = []
                    for i in range(len(current_layer)):
                        current_perceptron = current_layer[i]
                        current_input = input_per_perceptron[l][i]
                        current_activation = activation_per_perceptron[l][i]
                        derivate = current_perceptron.activation_function_derivate(current_input)

                        # some magic over here
                        magic_number_1 = [current_input] * len(current_gradient)
                        magic_number_2 = np.dot(magic_number_1, current_gradient)
                        current_delta.append(derivate*magic_number_2)
                    delta = [current_delta] + delta

                
                # update every weight in network
                for i in range(self.num_layers):
                    for j in range(len(self.network[i])):
                        update_value = learning_rate*activation_per_perceptron[i][j]*delta[i][j]
                        self.network[i][j].update_weights(update_value)

                for i in range(len(self.network)):
                    print(i)
                    for j in range(len(self.network[i])):
                        print(f"{self.network[i][j].weights} ", end="")
                a = input()
            if epoch == 100000:
                break

            epoch += 1
    
    def __evaluate_network(self, input_values: np.array) -> None:
        return None


