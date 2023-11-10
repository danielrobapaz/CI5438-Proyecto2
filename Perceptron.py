import numpy as np
from math import exp

class Perceptron():
    """
        Implementation of a Perceptron. 

        Constructor recieve a number of weights that the 
        perceptron will use.
    """
    def __init__(self, num_weights: int):
        self.weights = np.zeros(num_weights)

    def __input_function(self, input_values: np.ndarray) -> float:
        if len(input_values) != self.weights:
            raise Exception("Error: Len of input values not equal to len weights values. ")
        
        return np.dot(self.weights, input_values)
    
    """
        Compute the Perceptron's activations function.

        The activation function is the logistic function.
    """
    def activation_function(self, input_values: np.ndarray) -> float:
        if len(input_values) != self.weights:
            raise Exception("Error: Len of input values not equal to len weights values. ")
        
        x = self.__input_function(input_values)

        return 1/(1+exp(x))
    
    """
        Compute the derivate of the Perceptron's activation function.
    """
    def activation_function_derivate(self, input_values: np.ndarray) -> float:
        if len(input_values) != self.weights:
            raise Exception("Error: Len of input values not equal to len weights values. ")

        g = self.activation_function(input_values)

        return g*(1-g)