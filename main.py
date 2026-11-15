from src.image.exporter import save_image
from src.image.loader import load_image
from src.kernels.blur import create_box_blur_kernel, create_motion_blur_kernel
from src.kernels.edge_detection import (
    create_laplacian_kernel,
    create_prewitt_kernel,
    create_scharr_kernel,
    create_sobel_kernel,
)
from src.kernels.emboss import create_emboss_kernel
from src.kernels.gaussian import create_gaussian_kernel
from src.kernels.sharpen import create_sharpen_kernel, create_unsharp_kernel
from src.matrix import create_test_matrix
from src.operations.convolution import convolve_2d, convolve_3d
from src.operations.postprocessing import absolute, clip, min_max_normalize, relu
from src.visualization import display_matrices

# Constants
image_path = "assets/dog.jpg"
kernel = create_gaussian_kernel(size=5, sigma=2.0)
# kernel = create_sobel_kernel("x")
# kernel = create_sobel_kernel("y")
# kernel = create_prewitt_kernel("x")
# kernel = create_prewitt_kernel("y")
# kernel = create_scharr_kernel("x")
# kernel = create_scharr_kernel("y")
# kernel = create_laplacian_kernel("4-neighbor")
# kernel = create_laplacian_kernel("8-neighbor")
# kernel = create_box_blur_kernel(5)
# kernel = create_motion_blur_kernel(9, "diagonal")
# kernel = create_sharpen_kernel(amount=1.0)
# kernel = create_unsharp_kernel()
# kernel = create_emboss_kernel("south-east")


def main():
    image = load_image(image_path, grayscale=False)

    output = convolve_3d(image, kernel)
    print(f"Output: {output}")

    # processed_output = clip(output)
    processed_output = absolute(output)
    processed_output = min_max_normalize(output)
    # processed_output = relu(output)

    save_image("output.jpg", processed_output)

    matrices = [image, kernel, processed_output]
    titles = ["Input", "Kernel", "Output"]
    display_matrices(matrices, titles)


if __name__ == "__main__":
    main()
