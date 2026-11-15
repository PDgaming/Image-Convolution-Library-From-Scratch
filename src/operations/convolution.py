import numpy as np


def convolve_2d(image: np.ndarray, kernel: np.ndarray, padding: int = 0) -> np.ndarray:
    kernel_height, kernel_width = kernel.shape
    
    if padding < 0:
        raise ValueError("Padding must be a non-negative integer.")
    if padding == 0:
        padding_height = (kernel_height - 1) // 2
        padding_width = (kernel_width - 1) // 2
        image = np.pad(image, pad_width=(padding_height, padding_width), mode='constant', constant_values=0)
    else:
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

def convolve_3d(image: np.ndarray, kernel: np.ndarray, padding: int = 0) -> np.ndarray:
    image_height, image_width, channels = image.shape

    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    red_output = convolve_2d(red_channel, kernel, padding)
    green_output = convolve_2d(green_channel, kernel, padding)
    blue_output = convolve_2d(blue_channel, kernel, padding)

    output = np.stack((red_output, green_output, blue_output), axis=-1)
    
    return output