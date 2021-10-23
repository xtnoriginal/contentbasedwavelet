import numpy as np


def acceptance_criteria(query_sigma, database_sigma, beta=1 - (50 / 100)):
    return (query_sigma[0] * beta < database_sigma[0] < query_sigma[0] / beta) or (
            (query_sigma[1] * beta < database_sigma[1] < query_sigma[1] / beta) and (
            query_sigma[2] * beta < database_sigma[2] < query_sigma[2] / beta))


def dist(query_wc_1, database_wc_1):

    w11_query = query_wc_1[0]
    w11_database = database_wc_1[0]

    w22_query = query_wc_1[1]
    #todo
    return query_wc_1, database_image


def euclidean_dist(u ,v):
    return np.sqrt(np.sum(u-v)**2)