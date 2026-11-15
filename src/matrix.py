import numpy as np


def create_test_matrix(width: int, height: int) -> np.ndarray:
    array = np.zeros((height, width), dtype=np.float64)

    array[2:-2, 2:-2] = 255

    return array
