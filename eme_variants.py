"""
    Novin Tang
    novin.tang44@myhunter.cuny.edu
    CSCI 39534 Project 3 - Retinal Fundus Images
    Student 3 - EME and Its Variants
"""
import numpy as np
from PIL import Image
import os

def calculateEme(image, k1, k2):
    img_arr = np.array(image).astype(np.double)
    rows, cols = img_arr.shape
    block_rows, block_cols = rows // k1, cols // k2
    eme = 0

    for i in range(k1):
        for j in range(k2):
            row_start = i * block_rows
            row_end = (i + 1) * block_rows
            col_start = j * block_cols
            col_end = (j + 1) * block_cols

            block = img_arr[row_start:row_end, col_start:col_end]
            
            I_max = np.max(block)
            I_min = np.min(block)

            if I_min == 0:
                I_min = 1

            eme += 20 * np.log(I_max / I_min)
    
    eme /= (k1 * k2)
    return eme

def calculateEmee(image, a, k1, k2):
    img_arr = np.array(image).astype(np.double)
    rows, cols = img_arr.shape
    block_rows, block_cols = rows // k1, cols // k2
    emee = 0

    for i in range(k1):
        for j in range(k2):
            row_start = i * block_rows
            row_end = (i + 1) * block_rows
            col_start = j * block_cols
            col_end = (j + 1) * block_cols

            block = img_arr[row_start:row_end, col_start:col_end]
            
            I_max = np.max(block)
            I_min = np.min(block)

            if I_min == 0:
                I_min = 1

            emee += a * np.log(I_max / I_min)
    
    emee /= (k1 * k2)
    return emee

def calculateMichelsonContrast(image, k1, k2):
    img_arr = np.array(image).astype(np.double)
    rows, cols = img_arr.shape
    block_rows, block_cols = rows // k1, cols // k2
    visibility = 0

    for i in range(k1):
        for j in range(k2):
            row_start = i * block_rows
            row_end = (i + 1) * block_rows
            col_start = j * block_cols
            col_end = (j + 1) * block_cols

            block = img_arr[row_start:row_end, col_start:col_end]

            I_max = np.max(block)
            I_min = np.min(block)

            if I_min == 0:
                I_min = 1

            visibility += (I_max - I_min) / (I_max + I_min)

    return visibility

def calculateAme(image, k1, k2):
    img_arr = np.array(image).astype(np.double)
    rows, cols = img_arr.shape
    block_rows, block_cols = rows // k1, cols // k2
    ame = 0

    for i in range(k1):
        for j in range(k2):
            row_start = i * block_rows
            row_end = (i + 1) * block_rows
            col_start = j * block_cols
            col_end = (j + 1) * block_cols

            block = img_arr[row_start:row_end, col_start:col_end]

            I_max = np.max(block)
            I_min = np.min(block)

            # Conditional statement to avoid divide by zero or log(0) errors
            if (I_max + I_min) == 0 or (I_max - I_min) <= 0:
                continue
            else:
                ame += np.log((I_max - I_min) / (I_max + I_min))

    ame /= (k1 * k2 * -1)
    return ame

def calculateAmee(image, a, k1, k2):
    img_arr = np.array(image).astype(np.double)
    rows, cols = img_arr.shape
    block_rows, block_cols = rows // k1, cols // k2
    amee = 0

    for i in range(k1):
        for j in range(k2):
            row_start = i * block_rows
            row_end = (i + 1) * block_rows
            col_start = j * block_cols
            col_end = (j + 1) * block_cols

            block = img_arr[row_start:row_end, col_start:col_end]

            I_max = np.max(block)
            I_min = np.min(block)

            # Conditional statement to avoid divide by zero or log(0) errors
            if (I_max + I_min) == 0 or (I_max - I_min) <= 0:
                continue
            else:
                amee += np.log(((I_max - I_min) / (I_max + I_min)) ** a)

    amee /= (k1 * k2 * -1)
    return amee

a = 0.5
k1 = 10
k2 = 10

# Iterate through all images in `contrast_images/piecewise` directory to calculate values
enhanced_path = "contrast_images/piecewise"
print("-----Enhanced Images-----")
for filename in os.listdir(enhanced_path):
    img = Image.open(enhanced_path + "/" + filename)
    eme = calculateEme(img, k1, k2)
    emee = calculateEmee(img, a, k1, k2)
    visibility = calculateMichelsonContrast(img, k1, k2)
    ame = calculateAme(img, k1, k2)
    amee = calculateAmee(img, a, k1, k2)

    print(f"{filename}: {round(eme, 2)}, {round(emee, 2)}, {round(visibility, 2)}, {round(ame, 2)}, {round(amee, 2)}")

# Iterate through all images in `images/grayscale` directory to calculate values
original_path = "images/grayscale"
print("-----Grayscale Images-----")
for filename in os.listdir(original_path):
    img = Image.open(original_path + "/" + filename)
    eme = calculateEme(img, k1, k2)
    emee = calculateEmee(img, a, k1, k2)
    visibility = calculateMichelsonContrast(img, k1, k2)
    ame = calculateAme(img, k1, k2)
    amee = calculateAmee(img, a, k1, k2)

    print(f"{filename}: {round(eme, 2)}, {round(emee, 2)}, {round(visibility, 2)}, {round(ame, 2)}, {round(amee, 2)}")