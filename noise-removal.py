"""
    Jesse Han (Acting as Student 1)
    jesse.han53@myhunter.cuny.edu
    CSCI 39534 Project 3 - Retinal Fundus Images
    Noise Removal Portion
    Resources: Kaggle, for Images
               https://www.kaggle.com/datasets/kssanjaynithish03/retinal-fundus-images/data
               Researchgate.net, for 5x5 Gaussian Filter
               https://www.researchgate.net/figure/Discrete-approximation-of-the-Gaussian-kernels-3x3-5x5-7x7_fig2_325768087
               Sharpening Images
               https://ai.stanford.edu/~syyeung/cvweb/tutorial1.html
               calculate_EME.m, given from Lab 4
               
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import os

image_dir = 'images/original/'
grayscale_dir = 'images/grayscale/'
output_dir = 'student-1/'

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

# Applies a filter from a 2D array.
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

# Entropy Measure of Enhancement. Reused from Lab 4. Sourced from the given calculcate_EME.m file
def eme(image_file, split=10):
    img = image_file.copy()
    pixels = img.load()
    width = img.size[0]
    height = img.size[1]

    # This code uses the given calculate_EME.m as a reference.
    EME = 0
    block_width = width // split
    block_height = height // split

    for k in range(1, split):
        for l in range(1, split):
            width_start = (k-1) * block_width 
            width_end = k * block_width
            height_start = (l-1) * block_height
            height_end = l * block_height

            if width_end > width:
                width_end = width
            if height_end > height:
                height_end = height
            i_max = 0
            i_min = 255
            for i in range(width_start, width_end):
                for j in range(height_start, height_end):
                    i_max = max(i_max, pixels[i,j])
                    i_min = min(i_min, pixels[i,j])
            i_min = 1 if i_min == 0 else i_min
            EME += 20 * math.log(i_max / i_min)
    EME = EME / (split * split)
    return EME

# First part reused from Lab 4.
def linear_contrast_stretching(image_obj, t=127):
    image = image_obj.copy()
    pixels = image.load()
    width = image.size[0]
    height = image.size[1]
    output = Image.new(mode='L', size=(width, height))
    output_pixels = output.load()
    i_min = 255
    i_max = 0

    
    for i in range(width):
        for j in range(height):
            i_min = min(i_min, pixels[i,j])
            i_max = max(i_max, pixels[i,j])

    for i in range(width):
        for j in range(height):
            if pixels[i,j] > t:
                output_pixels[i,j] = (int) ((pixels[i,j] - t + 1) * ((float) (255 - t + 1) / (i_max - t + 1)) + t + 1)
            else:
                if (t - i_min) == 0:
                    output_pixels[i,j] = (int) ((pixels[i,j] - i_min) * ((float) (t)))
                else:
                    output_pixels[i,j] = (int) ((pixels[i,j] - i_min) * ((float) (t) / (t - i_min)))
    return output

start_time = time.time()
# Creating dict with {name: eme} as its entries
image_files = dict.fromkeys(os.listdir(image_dir), [0] * 256)
optimal_eme = dict.fromkeys(image_files, (-1, -1.0))

print(f'Fetching from {image_dir}')

for image_file in image_files:
    # Grayscale + EME of original
    image = Image.open(image_dir + image_file).convert('L')
    image.save(f'{grayscale_dir}{image_file}')
    image_files[image_file][0] = eme(image)

    # Applying linear contrast stretching with 
    for i in range(1,256):
        contrast_image = linear_contrast_stretching(image, i)

        contrast_eme = eme(contrast_image)
        image_files[image_file][i] = contrast_eme
        
        # contrast_image.save(f"{output_dir}{image_file.split('.')[0]}/{str(i)}_{image_file}")
        contrast_image.close()
    print(f"{image_file} base EME: {image_files[image_file][0]}")
    optimal_t = image_files[image_file].index(max(image_files[image_file]))
    optimal_eme[image_file] = (optimal_t, max(image_files[image_file]))
    contrast_image = linear_contrast_stretching(image, optimal_t).save(f"{output_dir}optimal-{str(optimal_t)}_{image_file}")
        
print(f'Process took: {time.time() - start_time:.4f} seconds!')
print(optimal_eme)
