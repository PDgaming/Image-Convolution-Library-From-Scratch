import numpy as np
import math


def create_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("Size must be an odd integer.")

    if sigma <= 0:
        raise ValueError("Sigma must be a positive number.")

    matrix = np.zeros((size, size))

    center_coordinate = tuple(dim // 2 for dim in matrix.shape)

    for i in range(size):
        for j in range(size):
            x = i - center_coordinate[0]
            y = j - center_coordinate[1]

            # Calculate the 2D Gaussian function
            amplitude = 1 / (2 * math.pi * sigma**2)
            exponential_function = math.exp(-(x**2 + y**2) / (2 * sigma**2))
            gaussian_value = amplitude * exponential_function

            matrix[i, j] = gaussian_value

    matrix = matrix / np.sum(matrix)

    return matrix
