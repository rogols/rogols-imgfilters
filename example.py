from imgfilters.filters import (
    grayscale,
    add_noise,
    gaussian_smoothing,
    denoise,
)
from imgfilters.file_io import read_image, save_image


image = read_image("data/astronaut.jpg")


image_grayscale = grayscale(image)
save_image(image_grayscale, "generated/astronaut-grayscale.jpg")


image_noisy = add_noise(image, amount=0.15)
save_image(image_noisy, "generated/astronaut-noisy.jpg")


image_smoothed = gaussian_smoothing(image_noisy, sigma=5)
save_image(image_smoothed, "generated/astronaut-smoothed.jpg")


image_denoised = denoise(image_noisy, magic_factor=1.2)
save_image(image_denoised, "generated/astronaut-denoised.jpg")
