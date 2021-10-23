import cv2
import pywt


def read_image(file_name):
    img = cv2.imread(file_name)
    return resize(img)

def read_database():
    for i in range(2000,2157):
        read_image('database/'+str(i)+'00.jpg')


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
