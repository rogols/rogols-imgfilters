import numpy as np
from skimage import io, img_as_float


def read_image(file_name):
    image = io.imread(file_name)
    return img_as_float(image)


def save_image(image, file_name):
    io.imsave(file_name, (image * 255).astype(np.uint8))
