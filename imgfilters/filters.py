import numpy as np
from skimage.filters import gaussian
from skimage import color, util


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
