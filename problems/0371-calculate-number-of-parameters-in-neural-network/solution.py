def calculate_parameters(layers: list[dict]) -> int:
    total_params = 0
    for layer in layers:
        layer_type = layer['type']
        bias = layer.get('bias', True)  # Use parentheses () here
        
        if layer_type == 'dense':
            input_size = layer['input_size']
            output_size = layer['output_size']
            weights = input_size * output_size
            biases = output_size if bias else 0
            total_params += weights + biases
            
        elif layer_type == 'conv2d':
            in_channels = layer['in_channels']
            out_channels = layer['out_channels']
            kernel_size = layer['kernel_size']
            weights = out_channels * in_channels * kernel_size * kernel_size
            biases = out_channels if bias else 0
            total_params += weights + biases
            
    return total_params