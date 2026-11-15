import numpy as np

def clip(matrix: np.ndarray) -> np.ndarray:
    return np.clip(matrix, 0, 255)

def absolute(matrix: np.ndarray) -> np.ndarray:
    return np.abs(matrix)

def min_max_normalize(matrix: np.ndarray) -> np.ndarray:
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    return (matrix - min_val) / (max_val - min_val) * 255

def relu(matrix: np.ndarray) -> np.ndarray:
    return np.maximum(0, matrix)