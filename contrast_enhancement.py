"""
    Erika Lin
    Erika.lin25@myhunter.cuny.edu
    CSCI 39534 Project 3 - Retinal Fundus Images
    Person 2 - Contrast Enhancement Pipeline
    Resources: Images - Kaggle
               https://www.kaggle.com/datasets/kssanjaynithish03/retinal-fundus-images/data

    t_optimum (image 1 - 11): 103, 183, 112, 97, 97, 121, 112, 74, 91, 101, 115
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Stretch before saving to to ensure values are in [0, 255]
def stretch_contrast(image):
    I_min, I_max = image.min(), image.max()
    if I_max - I_min == 0:
        return np.zeros_like(image)
    stretched = (image - I_min) * 255.0 / (I_max - I_min)
    return stretched.astype(np.uint8)

# t_optimum dictionary from student 1
t_optimum_dict = {
    "01": 103, "02": 183, "03": 112, "04": 97, "05": 97, "06": 121,
    "07": 112, "08": 74, "09": 91, "10": 101, "11": 115
}

input_folder = 'images/original/'
for filename in sorted(os.listdir(input_folder)):
    if filename.endswith(('.jpeg', '.jpg', '.png')):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)
        image_np = np.array(image)
        image_number = filename.split('_')[0]
        
        # Convert to grayscale
        gray = 0.299 * image_np[:, :, 0] + 0.587 * image_np[:, :, 1] + 0.114 * image_np[:, :, 2]
        gray = gray.astype(np.uint8)

        # Normalize each color channel
        epsilon = 1e-10
        added_rgb = (image_np[:, :, 0] + image_np[:, :, 1] + image_np[:, :, 2]).astype(np.float32) + epsilon
        r_normalized = image_np[:, :, 0].astype(np.float32) / added_rgb
        g_normalized = image_np[:, :, 1].astype(np.float32) / added_rgb
        b_normalized = image_np[:, :, 2].astype(np.float32) / added_rgb

        # Combine for single-channel normalized grayscale
        normalized_gray = (0.299 * r_normalized + 0.587 * g_normalized + 0.114 * b_normalized)
        normalized_gray = np.nan_to_num(normalized_gray, nan=0.0, posinf=1.0, neginf=0.0)
        normalized_gray = np.clip(normalized_gray, 0, 1)
        normalized_image = (normalized_gray * 255).astype(np.uint8)

        # Compute difference
        gray_float = gray.astype(np.float32) / 255.0
        normalized_image_float = normalized_image.astype(np.float32) / 255.0
        diff = np.abs(gray_float - normalized_image_float)

        # Gamma correction and alpha blending
        gamma = 2
        gamma_diff = np.power(diff, gamma)
        alpha = 0.5
        blended = alpha * gamma_diff + (1 - alpha) * gray_float
        blended_image = np.clip(blended * 255, 0, 255).astype(np.uint8)

        # Piecewise contrast enhancement
        t_optimum = t_optimum_dict.get(image_number)
        I_min = np.min(blended_image)
        I_max = np.max(blended_image)
        piecewise_image = blended_image.copy()

        # Loops through every pixel and applies a piecewise contrast enhancement
        for i in range(blended_image.shape[0]):
            for j in range(blended_image.shape[1]):
                pixel = blended_image[i, j]
                # if pixel intensity <= (darker)
                if pixel <= t_optimum:
                    piecewise_image[i, j] = int((pixel - I_min) * ((t_optimum - I_min) / (t_optimum - I_min)) + I_min)
                # else if pixel intensity > (brighter)
                else:
                    value = (pixel - t_optimum + 1) * ((I_max - t_optimum + 1) / (I_max - t_optimum + 1)) + t_optimum + 1
                    piecewise_image[i, j] = np.clip(int(value), 0, 255)

        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        ax[0].imshow(gray, cmap="gray")
        ax[0].set_title("Grayscale Image")
        ax[0].axis("off")

        ax[1].imshow(normalized_image, cmap="gray")
        ax[1].set_title("Normalized Grayscale Image")
        ax[1].axis("off")

        ax[2].imshow(blended_image, cmap="gray")
        ax[2].set_title("Blended (Gray + Gamma)")
        ax[2].axis("off")

        ax[3].imshow(piecewise_image, cmap="gray")
        ax[3].set_title("Piecewise Image")
        ax[3].axis("off")

        plt.tight_layout()
        plt.show()

        """
        base_name = os.path.splitext(filename)[0]
        Image.fromarray(stretch_contrast(gray)).save(f'contrast_images/grayscale/{base_name}_gray.jpeg')
        Image.fromarray(stretch_contrast(normalized_image)).save(f'contrast_images/normalized/{base_name}_normalized.jpeg')
        Image.fromarray(stretch_contrast(blended_image)).save(f'contrast_images/alpha_blend/{base_name}_alpha.jpeg')
        Image.fromarray(stretch_contrast(piecewise_image)).save(f'contrast_images/piecewise/{base_name}_piecewise.jpeg')
        """
