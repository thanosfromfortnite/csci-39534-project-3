"""
    Jesse Han
    jesse.han53@myhunter.cuny.edu
    CSCI 39534 Project 3 - Retinal Fundus Images
    Noise Removal Portion
    Resources: Kaggle, for Images
               https://www.kaggle.com/datasets/kssanjaynithish03/retinal-fundus-images/data
               Researchgate.net, for 5x5 Gaussian Filter
               https://www.researchgate.net/figure/Discrete-approximation-of-the-Gaussian-kernels-3x3-5x5-7x7_fig2_325768087
               Sharpening Images
               https://ai.stanford.edu/~syyeung/cvweb/tutorial1.html
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import os

image_dir = 'images/original/'
grayscale_dir = 'images/grayscale/'
output_dir = 'images/filtered/'

# Averaging Filter
a_filter = [[1.0 / 9.0] * 3] * 3
# Larger Average Filter
a_filter_large = [[1.0 / 49] * 7] * 7
# Gaussian Filter
g_filter = [
    [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
    [2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0],
    [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0]
]
# Larger Gaussian Filter
g_filter_large = [
    [1.0/273, 4.0/273, 7.0/273, 4.0/273, 1.0/273],
    [4.0/273, 16.0/273, 26.0/273, 16.0/273, 4.0/273],
    [7.0/273, 26.0/273, 41.0/273, 26.0/273, 7.0/273],
    [4.0/273, 16.0/273, 26.0/273, 16.0/273, 4.0/273],
    [1.0/273, 4.0/273, 7.0/273, 4.0/273, 1.0/273]
]

# Sharpened Filter
s_filter = [
    [-1.0/9, -1.0/9, -1.0/9],
    [-1.0/9, 2.0 - 1.0/9, -1.0/9],
    [-1.0/9, -1.0/9, -1.0/9]
]

# Extra Sharpened Filter
es_filter = [
    [-2.0/9, -2.0/9, -2.0/9],
    [-2.0/9, 3.0 - 2.0/9, -2.0/9],
    [-2.0/9, -2.0/9, -2.0/9]
]

def filter_2d_obj(image_obj, filter_2d):
    image_pixels = image.load()
    output = Image.new(mode='L', size=(image.size[0], image.size[1]))
    output_pixels = output.load()

    for i in range(image.size[0]):
        for j in range(image.size[1]):
            filter_sum = 0.0
            for k in range(len(filter_2d)):
                for l in range(len(filter_2d[0])):
                    if i + k >= 0 and i + k < image.size[0] and j + l >= 0 and j + l < image.size[1]:
                        filter_sum += image_pixels[i + k, j + l] * filter_2d[k][l]
            output_pixels[i,j] = (int) (filter_sum)
    return output


start_time = time.time()
print(f'Fetching from {image_dir}')
for image_file in os.listdir(image_dir):
    image = Image.open(image_dir + image_file).convert('L')
    image.save(grayscale_dir + image_file)

    image = filter_2d_obj(image, es_filter)
    image = filter_2d_obj(image, g_filter)
    image = filter_2d_obj(image, es_filter)

    image.save(output_dir + image_file)
    image.close()
print(f'Process took: {time.time() - start_time:.4f} seconds!')
