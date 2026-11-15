import numpy as np


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    output = np.zeros((output_height, output_width), dtype=np.float64)

    kernel = np.flip(kernel)

    for y in range(output_height):
        for x in range(output_width):
            region = image[y : y + kernel_height, x : x + kernel_width]
            output[y, x] = np.sum(region * kernel)

    return output
