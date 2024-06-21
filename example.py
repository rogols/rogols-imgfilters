import numpy as np
from skimage import io, img_as_float, color, util
from skimage.filters import gaussian


def read_image(file_name):
    image = io.imread(file_name)
    return img_as_float(image)


def save_image(image, file_name):
    io.imsave(file_name, (image * 255).astype(np.uint8))


def grayscale(image):
    return color.rgb2gray(image)


def add_noise(image, amount):
    noisy_image = util.random_noise(image, mode="s&p", amount=amount)
    return noisy_image


def gaussian_smoothing(image, sigma=10):
    smoothed_image = np.zeros_like(image)
    for i in range(image.shape[2]):
        smoothed_image[:, :, i] = gaussian(
            image[:, :, i], sigma=sigma, mode="constant", cval=0.0
        )
    return smoothed_image


image = read_image("data/astronaut.jpg")


image_grayscale = grayscale(image)
save_image(image_grayscale, "generated/astronaut-grayscale.jpg")


image_noisy = add_noise(image, amount=0.15)
save_image(image_noisy, "generated/astronaut-noisy.jpg")


image_smoothed = gaussian_smoothing(image_noisy, sigma=5)
save_image(image_smoothed, "generated/astronaut-smoothed.jpg")
