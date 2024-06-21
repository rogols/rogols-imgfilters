import numpy as np
from skimage.filters import gaussian
from skimage.restoration import denoise_nl_means, estimate_sigma
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


def denoise(image, magic_factor=0.6):
    # estimate the noise standard deviation from the noisy image
    sigma_est = np.mean(estimate_sigma(image, channel_axis=-1))

    # keyword arguments for the filter
    patch_kw = dict(
        patch_size=5,
        patch_distance=6,
        channel_axis=-1,
    )

    image_denoised = denoise_nl_means(
        image, h=magic_factor * sigma_est, sigma=sigma_est, fast_mode=True, **patch_kw
    )

    return image_denoised
