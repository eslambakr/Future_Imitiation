from skimage.util import random_noise
import numpy as np
from scipy import misc


def apply_noise(img, ratio, mode='s&p', scale=1):
    """
    :param img:
    :param ratio:
    :param mode:
    :param scale: the noise function by default normalise the iamge between 0 and 1. so set the scale to 255 if you want.
    :return:
    """
    img = random_noise(img, mode=mode, amount=ratio) * scale
    return img


def main():
    path = '/media/user/research_data1/test.png'
    img = misc.imread(path)/255
    print(img)
    img = apply_noise(img, 1)
    print(img)
    misc.imsave('test.png', img)


if __name__ == '__main__':
    main()
