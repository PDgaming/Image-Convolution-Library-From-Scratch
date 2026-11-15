import numpy as np


def create_emboss_kernel(direction: str = "south-east") -> np.ndarray:
    kernels = {
        "north-west": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float64),
        "south-east": np.array([[2, 1, 0], [1, 1, -1], [0, -1, -2]], dtype=np.float64),
        "north-east": np.array([[0, -1, -2], [1, 1, -1], [2, 1, 0]], dtype=np.float64),
        "south-west": np.array([[0, 1, 2], [-1, 1, 1], [-2, -1, 0]], dtype=np.float64),
    }

    if direction not in kernels:
        raise ValueError(
            "direction must be one of: " + ", ".join(sorted(kernels.keys()))
        )

    return kernels[direction]
