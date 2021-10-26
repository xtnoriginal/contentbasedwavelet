import cv2 as cv2
import numpy

import image_processing
import processing
import wavelet

if __name__ == '__main__':
    # Enter query image and  database folder
    # query_image_filename = input('Enter Query Image :')
    # folder_database = input('Enter Database Folder :')

    folder_database = 'database'
    query_image_filename = 'image/z.jpg'

    query_image = image_processing.get_query_image(query_image_filename)
    database = image_processing.read_database(folder_database)

    results = processing.three_stage_comparison(query_image, database)

    image_processing.show(query_img=query_image, results=results)




