import numpy as np
import PIL.Image as pil


def load_image(path: str, grayscale: bool = True) -> np.ndarray:
    image = pil.open(path)
    if grayscale:
        image = image.convert("L")
    else:
        image = image.convert("RGB")

    image_array = np.array(image)

    return image_array