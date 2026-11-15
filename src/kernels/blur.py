import numpy as np


def create_box_blur_kernel(size: int) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("Size must be an odd integer.")
    if size < 1:
        raise ValueError("Size must be at least 1.")

    return np.ones((size, size), dtype=np.float64) / (size * size)


def create_motion_blur_kernel(size: int, direction: str = "horizontal") -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("Size must be an odd integer.")
    if size < 1:
        raise ValueError("Size must be at least 1.")

    kernel = np.zeros((size, size), dtype=np.float64)
    center = size // 2

    if direction == "horizontal":
        kernel[center, :] = 1.0
    elif direction == "vertical":
        kernel[:, center] = 1.0
    elif direction == "diagonal":
        np.fill_diagonal(kernel, 1.0)
    elif direction == "anti-diagonal":
        np.fill_diagonal(np.fliplr(kernel), 1.0)
    else:
        raise ValueError(
            "direction must be 'horizontal', 'vertical', 'diagonal', or 'anti-diagonal'"
        )

    return kernel / np.sum(kernel)
