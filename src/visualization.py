import matplotlib.pyplot as plt
import numpy as np


def _prepare_for_display(matrix):
    matrix = np.asarray(matrix)
    if matrix.ndim == 2:
        if np.issubdtype(matrix.dtype, np.floating) and matrix.max(initial=0) > 1:
            return matrix, 0, 255
        return matrix, None, None
    if matrix.ndim == 3:
        if matrix.dtype != np.uint8:
            matrix = np.clip(matrix, 0, 255)
            if np.issubdtype(matrix.dtype, np.floating):
                matrix = matrix / 255.0
            else:
                matrix = matrix.astype(np.float64) / 255.0
        return matrix, None, None
    return matrix, None, None


def display_matrices(matrices: list, titles: list) -> None:
    count = len(matrices)

    fig, axes = plt.subplots(1, count, figsize=(5 * count, 5))
    if count == 1:
        axes = [axes]

    for matrix, title, ax in zip(matrices, titles, axes):
        matrix, vmin, vmax = _prepare_for_display(matrix)
        if matrix.ndim == 2:
            image = ax.imshow(matrix, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
            fig.colorbar(image, ax=ax)
        else:
            image = ax.imshow(matrix, interpolation="nearest")
        ax.set_title(title)

    plt.tight_layout()
    plt.show()