"""
    Jesse Han
    jesse.han53@myhunter.cuny.edu
    CSCI 39534 Project 3 - Retinal Fundus Images
    Noise Removal Portion
    Resources: Kaggle, for Images
               https://www.kaggle.com/datasets/kssanjaynithish03/retinal-fundus-images/data
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import os

image_dir = 'images/original/'
grayscale_dir = 'images/grayscale/'
output_dir = 'images/filtered/'

g_filter = [
    [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
    [2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0],
    [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0]
]

def filter_2d(image_file, filter_2d):
    
    image = Image.open(grayscale_dir + image_file)
    output_size = (image.size[0] + ((len(filter_2d) // 2) * 2), image.size[1] + ((len(filter_2d) // 2) * 2))
    output = Image.new(mode='L', size=output_size)
    image_pixels = image.load()
    output_pixels = output.load()

    for i in range(output_size[0]):
        for j in range(output_size[1]):
            filter_sum = 0.0
            for k in range(len(filter_2d)):
                for l in range(len(filter_2d[0])):
                    if i + k >= 0 and i + k < image.size[0] and j + l >= 0 and j + l < image.size[1]:
                        filter_sum += image_pixels[i + k, j + l] * filter_2d[k][l]
            output_pixels[i,j] = (int) (filter_sum)
    output.save(output_dir + image_file)
    
    image.close()
    output.close()

print(f'Fetching from {image_dir}')
for image_file in os.listdir(image_dir):
    image = Image.open(image_dir + image_file).convert('L')
    image.save(grayscale_dir + image_file)
    filter_2d(image_file, g_filter)
    image.close()
