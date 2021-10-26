import numpy as np

from image_processing import Image
from queue import PriorityQueue


def three_stage_comparison(query_image: Image, database):
    '''
    Implements the 3 stage comparisons to search  for  images
    :param query_image:
    :param database:
    :return:
    '''
    print(len(database))
    min_heap = PriorityQueue()
    count = 0
    for image in database:
        if acceptance_criteria(query_image.sigma, image.sigma):
            if euclidean_dist(min_max(image.wc_5), min_max(query_image.wc_5)) < 5:
                distance = dist2(image.wc_i, query_image.wc_i)
                min_heap.put((distance, image))

    res = []
    for i in range(20):
        if min_heap.empty():
            break

        res.append(min_heap.get())

    return res


def min_max(arr):
    # to be deleted
    mn = np.min(arr)
    mx = np.max(arr)
    return (arr - mn) * (1.0 / (mx - mn))


def acceptance_criteria(query_sigma, database_sigma, beta=1 - (50 / 100)):
    return (query_sigma[0] * beta < database_sigma[0] < query_sigma[0] / beta) or (
            (query_sigma[1] * beta < database_sigma[1] < query_sigma[1] / beta) and (
            query_sigma[2] * beta < database_sigma[2] < query_sigma[2] / beta))


def dist(query_wc_1, database_wc_1):
    w11_query = query_wc_1[0]
    w11_database = database_wc_1[0]

    w12_query = query_wc_1[1]['ad']
    w12_database = database_wc_1[1]['ad']

    w21_query = query_wc_1[1]['da']
    w21_database = database_wc_1[1]['da']

    w22_query = query_wc_1[1]['dd']
    w22_database = database_wc_1[1]['dd']

    w11 = w11_query * np.sum(np.subtract(w21_query, w11_database))
    w12 = w12_query * np.sum(np.subtract(w12_query, w12_database))
    w21 = w21_query * np.sum(np.subtract(w21_query, w21_database))
    w22 = w22_query * np.sum(np.subtract(w22_query, w22_database))

    return w11 + w12 + w21 + w22


def euclidean_dist(u, v):
    return np.sqrt(np.sum(u - v) ** 2)


def dist2(idx_WCi, WCi):
    """
    Computes the distance between the wavelet feature vectors
    :param idx_WCi: Indexed image 2 last wavelet features
    :param wci: Query image 2 last wavelet features
    :return: Euclidean distance
    """

    W11 = W12 = W21 = W22 = WC1 = WC2 = WC3 = 1
    wci = np.array([WC1, WC2, WC3])

    idx_W11 = idx_WCi[0]
    WC11 = WCi[0]

    idx_W12 = idx_WCi[1]['da']
    WC12 = WCi[1]['da']

    idx_W21 = idx_WCi[1]['ad']
    WC21 = WCi[1]['ad']

    idx_W22 = idx_WCi[1]['dd']
    WC22 = WCi[1]['dd']

    return W11 * np.sum(wci * euclidean_dist(WC11, idx_W11)) \
           + W12 * np.sum(wci * euclidean_dist(WC12, idx_W12)) \
           + W21 * np.sum(wci * euclidean_dist(WC21, idx_W21)) \
           + W22 * np.sum(wci * euclidean_dist(WC22, idx_W22))
