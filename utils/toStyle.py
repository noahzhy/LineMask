import os
import glob
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


BG_COLOR = 209
BG_SIGMA = 3
MONOCHROME = 1


def add_noise(img, sigma=BG_SIGMA):
    """
    Adds noise to the existing image
    """
    width, height, ch = img.shape
    n = noise(width, height, sigma=sigma)
    img = img + n
    return img.clip(0, 255)


def noise(width, height, ratio=1, sigma=BG_SIGMA):
    """
    The function generates an image, filled with gaussian nose. If ratio parameter is specified,
    noise will be generated for a lesser image and then it will be upscaled to the original size.
    In that case noise will generate larger square patterns. To avoid multiple lines, the upscale
    uses interpolation.

    :param ratio: the size of generated noise "pixels"
    :param sigma: defines bounds of noise fluctuations
    """
    mean = 0
    # assert width % ratio == 0, "Can't scale image with of size {} and ratio {}".format(width, ratio)
    # assert height % ratio == 0, "Can't scale image with of size {} and ratio {}".format(height, ratio)

    h = int(height / ratio)
    w = int(width / ratio)

    result = np.random.normal(mean, sigma, (w, h, MONOCHROME))
    if ratio > 1:
        result = cv2.resize(result, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
    return result.reshape((width, height, MONOCHROME))


def texture(image, sigma=BG_SIGMA, turbulence=2):
    """
    Consequently applies noise patterns to the original image from big to small.

    sigma: defines bounds of noise fluctuations
    turbulence: defines how quickly big patterns will be replaced with the small ones. The lower
    value - the more iterations will be performed during texture generation.
    """
    result = image.astype(float)
    cols, rows, ch = image.shape
    ratio = cols
    while not ratio == 1:
        result += noise(cols, rows, ratio, sigma=sigma)
        ratio = (ratio // turbulence) or 1
    cut = np.clip(result, 0, 255)
    return cut.astype(np.uint8)


if __name__ == '__main__':
    # get all svg files
    svg_files = glob.glob('E:/dataset/charts/synth_charts/line/*.png')
    # random select one
    svg_file = random.choice(svg_files)
    # read png file
    img = cv2.imread(svg_file)
    # resize to 512x512
    img = cv2.resize(img, (512, 512))
    # save as before.png
    cv2.imwrite('before.png', img)
    # add noise and texture
    img = add_noise(texture(img, sigma=2, turbulence=4), sigma=3)
    # save
    cv2.imwrite('debug.png', img)


# def show(img):
#     cv2.imshow('img', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# def apply_effect(img):
#     # add guassian noise
#     img = np.array(img / 255, dtype=float)

#     noise = np.random.normal(0, 0.03, img.shape)
#     # add noise on image
#     img -= noise
#     # clip to [0, 1]
#     img = np.clip(img, 0, 1)
#     # convert to uint8
#     img = np.array(img * 255, dtype=np.uint8)
#     return img




# # main
# if __name__ == "__main__":
#     # get all svg files
#     svg_files = glob.glob('E:/dataset/charts/synth_charts/bar/*.png')
#     # random select one
#     svg_file = random.choice(svg_files)
#     # read png file
#     img = cv2.imread(svg_file)
#     # apply effect
#     img = apply_effect(img)
#     # show(img)
#     # save to file
#     cv2.imwrite('debug.png', img)
