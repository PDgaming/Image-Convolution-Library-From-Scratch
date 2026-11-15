import numpy as np


def clip(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 2:
        return np.clip(matrix, 0, 255)
    if matrix.ndim == 3:
        return np.clip(matrix, 0, 255).astype(np.uint8)
    raise ValueError("matrix must be 2D or 3D (H, W, C)")


def absolute(matrix: np.ndarray) -> np.ndarray:
    return np.abs(matrix)


def min_max_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 2:
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        if max_val - min_val == 0:
            return np.zeros_like(matrix)
        return (matrix - min_val) / (max_val - min_val) * 255

    if matrix.ndim == 3:
        result = np.empty_like(matrix, dtype=np.float64)
        for c in range(matrix.shape[2]):
            channel = matrix[:, :, c]
            min_val = np.min(channel)
            max_val = np.max(channel)
            if max_val - min_val == 0:
                result[:, :, c] = 0
            else:
                result[:, :, c] = (channel - min_val) / (max_val - min_val) * 255
        return result

    raise ValueError("matrix must be 2D or 3D (H, W, C)")


def relu(matrix: np.ndarray) -> np.ndarray:
    return np.maximum(0, matrix)
