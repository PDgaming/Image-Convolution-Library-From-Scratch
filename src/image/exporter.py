import numpy as np
import PIL.Image as pil


def _to_pil_image(matrix: np.ndarray) -> pil.Image:
    matrix = np.asarray(matrix)

    if matrix.ndim == 2:
        if np.issubdtype(matrix.dtype, np.floating):
            matrix = np.clip(matrix, 0, 255).astype(np.uint8)
        elif matrix.dtype != np.uint8:
            matrix = matrix.astype(np.uint8)
        return pil.fromarray(matrix, mode="L")

    if matrix.ndim == 3:
        if matrix.shape[2] == 1:
            matrix = matrix[:, :, 0]
            if np.issubdtype(matrix.dtype, np.floating):
                matrix = np.clip(matrix, 0, 255).astype(np.uint8)
            elif matrix.dtype != np.uint8:
                matrix = matrix.astype(np.uint8)
            return pil.fromarray(matrix, mode="L")

        if np.issubdtype(matrix.dtype, np.floating):
            matrix = np.clip(matrix, 0, 255).astype(np.uint8)
        elif matrix.dtype != np.uint8:
            matrix = matrix.astype(np.uint8)
        return pil.fromarray(matrix, mode="RGB")

    raise ValueError("matrix must be 2D or 3D (H, W) or (H, W, C)")


def save_image(path: str, matrix: np.ndarray) -> str:
    image = _to_pil_image(matrix)
    image.save(path)
    return path


def export_image(path: str, matrix: np.ndarray) -> str:
    return save_image(path, matrix)
