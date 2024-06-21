import numpy as np
from skimage.filters import gaussian
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import color, filters, img_as_float, util


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


def _apply_warhol_effect(binary_image, color1, color2):
    colored_image = np.zeros(
        (binary_image.shape[0], binary_image.shape[1], 3), dtype=np.uint8
    )
    colored_image[binary_image] = color1
    colored_image[~binary_image] = color2
    return colored_image


def warhol_effect(image):
    gray_image = grayscale(image)

    # apply threshold to create a binary image
    thresh = filters.threshold_otsu(gray_image)
    binary_image = gray_image > thresh

    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 192, 203),  # Pink
    ]

    for i, c in enumerate(colors):
        new_image = _apply_warhol_effect(binary_image, c, (255, 255, 255))
        new_image = img_as_float(new_image)

        if i == 0:
            combined_image = new_image
        else:
            combined_image = np.concatenate((combined_image, new_image), axis=1)

    return combined_image
