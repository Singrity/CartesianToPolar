import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def image_load(path):
    img = Image.open(path)
    img.load()
    data = np.asarray(img, dtype='int32')
    return data


def convert_to_polar(matrix):
    x_data = matrix[:, :, 0].astype("float")
    y_data = matrix[:, :, 1].astype("float")
    r = np.sqrt(np.power(x_data, 2), np.power(y_data, 2))
    alpha = np.arctan2(y_data, x_data)
    matrix[:, :, 0] = r
    matrix[:, :, 1] = alpha
    return matrix


def convert_to_certain(matrix):
    r = matrix[:, :, 0]
    alpha = matrix[:, :, 1]
    x = r * np.cos(alpha)
    y = r * np.sin(alpha)
    matrix[:, :, 0] = x
    matrix[:, :, 1] = y
    return matrix.astype('int32')


def rotate(polar_matrix, angle):
    polar_matrix[:, :, 0].astype('float')
    polar_matrix += angle
    return polar_matrix


def change_r(polar_matrix, amount):
    polar_matrix[:, :, 0].astype('float')
    polar_matrix += amount
    return polar_matrix


image_matrix = image_load('../data/example.png')
plt.imshow(image_matrix)
plt.show()
polar_image = convert_to_polar(image_matrix)
new_image = convert_to_certain(rotate(polar_image, 90))
plt.imshow(new_image)
plt.show()
new_image = convert_to_certain(change_r(polar_image, 90))
plt.imshow(new_image)
plt.show()





