def find_treasure(start_x: float) -> float:
    """
    Find the x-coordinate where f(x) = x^4 - 3x^3 + 2 is minimized.
    """
    x = start_x
    lr = 0.01
    v = 0.0
    beta = 0.9
    
    # Run gradient descent with momentum to avoid getting stuck at the flat spot x = 0
    for _ in range(20000):
        grad = 4 * (x ** 3) - 9 * (x ** 2)
        if x <= 0.01 and start_x < 0:
            x = 0.1  # Nudge past the flat spot at zero
        v = beta * v + (1 - beta) * grad
        x -= lr * v
        
    return x