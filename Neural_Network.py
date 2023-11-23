from Neuron_Layer import Neuron_Layer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Neural_Network():

    '''Implements a Neural network that operates
    with backpropagation and gradient descent.'''
    def __init__(self, 
                 dataset: pd.DataFrame,
                 dep_vars: [str],
                 ind_vars: [str],
                 hidden_layer_lengths: [int],
                 max_iter: int,
                 learning_rate: float = 0.01,
                 ) -> None:
        
        #Initialize class members
        self.dataset = dataset
        self.dep_vars = dep_vars
        self.ind_vars = ind_vars
        self.hidden_layer_lengths = hidden_layer_lengths
        self.layer_count = len(hidden_layer_lengths) + 2 #Account for input and output layers
        self.layers: np.ndarray = np.array([])
        self.max_iter = max_iter
        self.learning_rate = learning_rate

        #Create network structure with network layers
        for i in range(self.layer_count-1):
            if i == 0:
                #Case of the input layer
                self.layers = np.append(self.layers, Neuron_Layer(len(self.ind_vars), 0))
            else:
                #Case of hidden layers
                prev_layer_len = self.layers[i-1].get_neuron_count()
                self.layers = np.append(self.layers, Neuron_Layer(hidden_layer_lengths[i-1], prev_layer_len))

        #Append output layer
        self.layers = np.append(self.layers, Neuron_Layer(len(dep_vars), self.layers[-1].get_neuron_count()))

    def __set_input_layer(self, x: pd.Series) -> None:
        self.layers[0].act_values = pd.Series.to_numpy(x, copy=True)

    """
    Calculates the output of the network under current
    weights and biases.
    """
    def __calculate_output(self, input: pd.Series) -> np.ndarray:
        self.__set_input_layer(input)
        for i in range(1, len(self.layers)):    #1 dado que no se activa la capa de entrada
            #Activar la capa i
            self.layers[i].activate_layer(self.layers[i-1].get_activations())
        return self.layers[-1].get_activations() #Devolver activaciones de la capa de salida
    
    """
    Trains and tests the model under the provided dataset
    """
    def get_model(self, split_ratio: float = 0.8) -> None:
        x = self.dataset[self.ind_vars]
        y = self.dataset[self.dep_vars]

        x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=split_ratio, random_state=42)
        train_df = pd.concat([x_train, y_train], axis=1)
        test_df = pd.concat([x_test, y_test], axis=1)
        
        #Train the network
        self.__train_network(train_df, self.max_iter)

        #Show network tests
        failed_tests = 0
        test_num = 0

        training_error = []
        false_positive = 0
        false_negative = 0
        for (index, row) in test_df.iterrows():
            test_input = row[self.ind_vars]
            test_output = pd.Series.to_numpy(row[self.dep_vars])

            # Model output
            output = self.__calculate_output(test_input)
            
            # Get the index of the highest output neuron
            chosen_index = np.argmax(output) 

            # Get the index of the highest output expected
            ans_index = np.argmax(test_output)

            training_error.append(np.abs(output[chosen_index]-test_output[ans_index]))

            if chosen_index != ans_index:
                failed_tests += 1
            test_num += 1
        
        mean_error = np.mean(training_error)
        min_error = np.min(training_error)
        max_error = np.max(training_error)

        print(f"Mean of error: {mean_error}")
        print(f"Minimun error: {min_error}")
        print(f"Maximun error: {max_error}")
        print(f"Number of tests: {test_num}")
        print(f"Number of tests failed: {failed_tests}")

        return [mean_error, min_error, max_error]
    
    """
    Performs the backpropagation algorithm and updates the weights under
    a given training set.
    """
    def __train_network(self, train_set: pd.DataFrame, epochs: int) -> None:
        epoch = 0
        mean_error_per_epoch = []
        max_error_per_epoch = []
        min_error_per_epoch = []

        while epoch < epochs:
            current_epoch_error = []
            
            for (index, row) in train_set.iterrows():
                x = row[self.ind_vars]
                y = row[self.dep_vars]

                #Calculate network output
                hw = self.__calculate_output(x)
                error = y - hw

                current_epoch_error.append(np.abs(error[np.argmax(hw)]))

                #Update output layer delta
                self.layers[-1].update_delta_output(error)

                #Update weights and deltas for all layers
                for i in range(len(self.layers)-2, -1, -1):
                    #Get current layer
                    curr_layer = self.layers[i]
                    #Update delta for this layer using the next layer's delta and weights
                    self.layers[i].update_delta_hidden(self.layers[i+1])
                    #Update weights from current layer to next layer (W_j,k)
                    self.layers[i+1].update_weights(curr_layer.get_activations(), self.learning_rate)
            epoch += 1

            mean_error_per_epoch.append(np.mean(current_epoch_error))
            max_error_per_epoch.append(np.max(current_epoch_error))
            min_error_per_epoch.append(np.min(current_epoch_error))

        self.__plot_error(mean_error_per_epoch, 
                          max_error_per_epoch, 
                          min_error_per_epoch, 
                          "Error per epoch in training")

    """
    Plot the changes in the mean, minimun and maximun error of the neural network
    """
    def __plot_error(self,
                     mean_error: [float],
                     max_error: [float],
                     min_error: [float],
                     title: str) -> None:
        
        x = np.linspace(1, self.max_iter, num=self.max_iter)
        
        fig, axs = plt.subplots(3)
        fig.suptitle(title)
        axs[0].plot(x, mean_error)
        axs[1].plot(x, min_error)
        axs[2].plot(x, max_error)
        
        ylabels = ["Mean error", "Minimin error", "Maximun error"]
        for i in [0, 1, 2]:
            axs[i].set_xlabel("Epoch")
            axs[i].set_ylabel(ylabels[i])
        
        fig.tight_layout()
        fig.set_figwidth(12)
        fig.set_figheight(8)
        plt.show()