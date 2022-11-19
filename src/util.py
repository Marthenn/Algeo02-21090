import numpy as np

__image_dim = 80


def normalize_image_value(arr):
    return (255 * (arr - np.min(arr)) / np.ptp(arr)).astype(float)


def get_image_dim():
    return __image_dim


# TODO: taro euclidean distance disini @Marten