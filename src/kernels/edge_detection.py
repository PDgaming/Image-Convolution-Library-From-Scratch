import numpy as np


def create_sobel_kernel(axis: str = "x") -> np.ndarray:
    if axis == "x":
        return np.array(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1],
            ],
            dtype=np.float64,
        )
    if axis == "y":
        return np.array(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1],
            ],
            dtype=np.float64,
        )
    raise ValueError("axis must be 'x' or 'y'")


def create_prewitt_kernel(axis: str = "x") -> np.ndarray:
    if axis == "x":
        return np.array(
            [
                [-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1],
            ],
            dtype=np.float64,
        )
    if axis == "y":
        return np.array(
            [
                [-1, -1, -1],
                [0, 0, 0],
                [1, 1, 1],
            ],
            dtype=np.float64,
        )
    raise ValueError("axis must be 'x' or 'y'")


def create_scharr_kernel(axis: str = "x") -> np.ndarray:
    if axis == "x":
        return np.array(
            [
                [3, 0, -3],
                [10, 0, -10],
                [3, 0, -3],
            ],
            dtype=np.float64,
        )
    if axis == "y":
        return np.array(
            [
                [3, 10, 3],
                [0, 0, 0],
                [-3, -10, -3],
            ],
            dtype=np.float64,
        )
    raise ValueError("axis must be 'x' or 'y'")


def create_laplacian_kernel(variant: str = "8-neighbor") -> np.ndarray:
    if variant == "4-neighbor":
        return np.array(
            [
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0],
            ],
            dtype=np.float64,
        )
    if variant == "8-neighbor":
        return np.array(
            [
                [1, 1, 1],
                [1, -8, 1],
                [1, 1, 1],
            ],
            dtype=np.float64,
        )
    raise ValueError("variant must be '4-neighbor' or '8-neighbor'")
