from src.kernels.constant_kernels import (
    edge_detection_kernel,
    emboss_kernel,
    gaussian_kernel,
)
from src.matrix import create_test_matrix
from src.operations.convolution import convolve
from src.operations.postprocessing import absolute, clip, min_max_normalize, relu
from src.visualization import display_matrices

TEST_WIDTH = 15
TEST_HEIGHT = 15
kernel = gaussian_kernel



def main():
    array = create_test_matrix(TEST_WIDTH, TEST_HEIGHT)
    print(f"Input, {array}")

    output = convolve(array, kernel)
    print(f"Output: {output}")

    processed_output = min_max_normalize(output)

    matrices = [array, kernel, processed_output]
    titles = ["Input", "Kernel", "Output"]
    display_matrices(matrices, titles)


if __name__ == "__main__":
    main()
