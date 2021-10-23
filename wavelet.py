import warnings

import cv2
import pywt
import numpy as np


def wavelet_deb(img, level):
    return pywt.wavedecn(img, 'db1', level=level, axes=(0, 1))


def get_features(img):
    # wc_1 = wavelet_deb(img, 4)
    wc_i = _wavedecn(img, 'db1', 4)
    sigma = np.std(wc_i[0], axis=(0, 1))
    wc_5 = _wavedecn(img, 'db1', 5)

    return wc_i, sigma, wc_5


def get_wavelet_features(img, wavelet, level):
    """
    Computes the wavelet features.
    :param img: Source image
    :param wavelet: Wavelet type
    :param level: Decomposition level
    :return: 3-tuple with features
    """

    WCi = _wavedecn(img, wavelet, level)
    sigma_ci = np.std(WCi[0], axis=(0, 1))
    print(sigma_ci)
    l5_WCi = _wavedecn(img, wavelet, level + 1)
    return WCi[:2].copy(), sigma_ci.copy(), l5_WCi[0].copy()


def _wavedecn(data, wavelet, level):
    # https://github.com/PyWavelets/pywt/issues/396
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pywt.wavedecn(data, wavelet, level=level, axes=(0, 1))


def _show_wavelet_image(coeffs, level):
    # normalize each coefficient array independently for better visibility
    coeffs[0] /= np.abs(coeffs[0]).max()
    for detail_level in range(level):
        coeffs[detail_level + 1] = [d / np.abs(d).max() for d in coeffs[detail_level + 1]]
    arr, slices = pywt.coeffs_to_array(coeffs)
    _show_image(arr)


def _show_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
