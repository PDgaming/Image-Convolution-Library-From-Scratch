import numpy as np


def create_sharpen_kernel(amount: float = 1.0) -> np.ndarray:
    if amount < 0:
        raise ValueError("amount must be non-negative.")

    center = 1.0 + 4.0 * amount
    edge = -amount
    return np.array(
        [
            [0, edge, 0],
            [edge, center, edge],
            [0, edge, 0],
        ],
        dtype=np.float64,
    )


def create_unsharp_kernel() -> np.ndarray:
    return np.array(
        [
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1],
        ],
        dtype=np.float64,
    )
