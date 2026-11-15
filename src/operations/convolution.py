import numpy as np


def convolve(image: np.ndarray, kernel: np.ndarray, padding: int = 0) -> np.ndarray:
    kernel_height, kernel_width = kernel.shape
    
    if padding < 0:
        raise ValueError("Padding must be a non-negative integer.")
    if padding == 0:
        padding_height = (kernel_height - 1) // 2
        padding_width = (kernel_width - 1) // 2
        image = np.pad(image, pad_width=(padding_height, padding_width), mode='constant', constant_values=0)

    image = np.pad(image, padding, mode='constant', constant_values=0)    

    image_height, image_width = image.shape

    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    output = np.zeros((output_height, output_width), dtype=np.float64)

    kernel = np.flip(kernel)

    for y in range(output_height):
        for x in range(output_width):
            region = image[y : y + kernel_height, x : x + kernel_width]
            output[y, x] = np.sum(region * kernel)

    return output
