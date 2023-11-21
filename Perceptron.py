import numpy as np
from math import exp

class Perceptron():
    """
        Implementation of a Perceptron. 

        Constructor recieve a number of weights that the 
        perceptron will use.
    """
    def __init__(self, num_weights: int):
        self.weights = np.zeros(num_weights+1) 

    def input_function(self, input_values: np.ndarray) -> float:
        if len(input_values) != len(self.weights):
            raise Exception("Error: Len of input values not equal to len weights values. ")
        return np.dot(self.weights,input_values)
    
    """
        Compute the Perceptron's activations function.

        The activation function is the logistic function.
    """
    def activation_function(self, input_values: np.ndarray) -> float:
        if len(input_values) != len(self.weights):
            raise Exception("Error: Len of input values not equal to len weights values. ")
        
        x = self.input_function(input_values)

        return 1/(1+exp(-x))
    
    """
        Compute the derivate of the Perceptron's activation function.
    """
    def activation_function_derivate(self, input_value: np.ndarray) -> float:
        g = input_value
        return g*(1-g)