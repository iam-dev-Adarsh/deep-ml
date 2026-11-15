import numpy as np
import math

def digit_network_forward(image, weights, biases):
    """Run an image through a fully connected sigmoid network.
    Returns (output activations rounded to 4 decimals, predicted digit).
    """
    # Flatten the 2D image row by row into a 1D list
    a = [pixel for row in image for pixel in row]
    
    # Iterate through each layer's weights and biases
    for W, b in zip(weights, biases):
        # Compute W * a + b for each neuron in the current layer
        z = [sum(w_i * a_i for w_i, a_i in zip(row, a)) + bias_val for row, bias_val in zip(W, b)]
        # Apply sigmoid activation function
        a = [1 / (1 + math.exp(-val)) for val in z]
        
    # Round activations to 4 decimal places
    activations = [round(val, 4) for val in a]
    
    # Find the index of the most active output neuron
    digit = a.index(max(a))
    
    return activations, digit