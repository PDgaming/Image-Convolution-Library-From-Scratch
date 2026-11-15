from src.kernels.gaussian import create_gaussian_kernel
from src.matrix import create_test_matrix
from src.operations.convolution import convolve
from src.visualization import display_matrices

TEST_WIDTH = 15
TEST_HEIGHT = 15
KERNEL_SIZE = 3
GAUSSIAN_SIGMA = 1


def main():
    array = create_test_matrix(TEST_WIDTH, TEST_HEIGHT)
    print(f"Input, {array}")

    gaussian_kernel = create_gaussian_kernel(KERNEL_SIZE, GAUSSIAN_SIGMA)
    print(f"Kernel: {gaussian_kernel}")

    output = convolve(array, gaussian_kernel)
    print(f"Output: {output}")

    matrices = [array, gaussian_kernel, output]
    titles = ["Input", "Kernel", "Output"]
    display_matrices(matrices, titles)


if __name__ == "__main__":
    main()
