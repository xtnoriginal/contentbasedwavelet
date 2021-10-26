import os
import cv2
import numpy as np
import pywt

import wavelet


class Image:

    def __init__(self, file_name, wc_i, sigma, wc_5):
        self.file_name = file_name
        self.wc_i = wc_i
        self.sigma = sigma
        self.wc_5 = wc_5


def read_image(file_name):
    img = cv2.imread(file_name)
    return resize(img)


def get_query_image(file_name):
    '''
    Get the query image and calculate the wavelet features
    :param file_name:
    :return: query_image: Image
    '''
    img = read_image(file_name)
    wc_i, sigma, wc_5 = wavelet.get_wavelet_features(img, 'db1', 4)
    return Image(dir, wc_i, sigma, wc_5)


def read_database(folder):
    images = []
    for filename in os.listdir(folder):
        dir = os.path.join(folder, filename)
        img = read_image(dir)
        if img is not None:
            wc_i, sigma, wc_5 = wavelet.get_wavelet_features(img, 'db1', 4)
            images.append(Image(dir, wc_i, sigma, wc_5))

    return images


def resize(img):
    return cv2.resize(img, (128, 128), 0, 0)


def display(img):
    cv2.imshow('Images', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def change_color_space(img, max=255):
    b, g, r = cv2.split(img)
    c1 = (r + g + b) / 3
    c2 = (r + (max - b)) / 2
    c3 = (r + 2 * (max - g) + b) / 4

    return cv2.merge((c1, c2, c3))


def save_image(file_name, img):
    cv2.imwrite(file_name, img)


# ---------------------------------------------------------------------------------
def show(query_img, results):
    GALLERY_COLUMNS = 5
    SHOW_IMG_MAX = 20

    opencv_bg = 50
    n_results = len(results) if len(results) <= SHOW_IMG_MAX else SHOW_IMG_MAX
    remain = ((((n_results // GALLERY_COLUMNS) + 1) * GALLERY_COLUMNS) - n_results) % GALLERY_COLUMNS
    fill = [np.full(query_img.shape, opencv_bg, dtype=np.uint8) for _ in range(remain)]

    for _, e in results[:n_results]:
        print(type(cv2.imread(e.file_name)))
    to_show = [cv2.imread(e.file_name).tolist() for _, e in results[:n_results]]

    #g = gallery(np.array(to_show + fill), ncols=GALLERY_COLUMNS)

    '''cv2.imshow('Query image', query_img)
    cv2.moveWindow('Query image', 0, 0)
    cv2.imshow('Results', g)
    cv2.moveWindow('Results', 400, 0)
    cv2.waitKey(0)'''


def gallery(array, ncols=3):

    #nindex, height, width, intensity = array.shape
    nindex, height, width, intensity = 20,1200,1600,3
    nrows = nindex // ncols
    assert nindex == nrows * ncols
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1, 2)
              .reshape(height * nrows, width * ncols, intensity))
    
    return result
