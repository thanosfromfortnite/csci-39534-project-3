"""
    Erika Lin
    Erika.lin25@myhunter.cuny.edu
    CSCI 39534 Project 3 - Retinal Fundus Images
    Person 2 - Contrast Enhancement Pipeline
    Resources: Images - Kaggle
               https://www.kaggle.com/datasets/kssanjaynithish03/retinal-fundus-images/data

"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open('images/original/03_mild_dr.jpeg')
image = image.convert('RGB')
image_np = np.array(image)

# Convert to gray = 0.299(R) + 0.587(G) + 0.114(B)
gray = 0.299 * image_np[:, :, 0] + 0.587 * image_np[:, :, 1] + 0.114 * image_np[:, :, 2]
gray = gray.astype(np.uint8)

# Normalize each color channel
epsilon = 1e-10
added_rgb = (image_np[:, :, 0] + image_np[:, :, 1] + image_np[:, :, 2]).astype(np.float32) + epsilon

r_normalized = image_np[:, :, 0].astype(np.float32) / added_rgb # (r/r+g+b)
g_normalized = image_np[:, :, 1].astype(np.float32) / added_rgb # (g/r+g+b)
b_normalized = image_np[:, :, 2].astype(np.float32) / added_rgb # (b/r+g+b)

# Convert to grayscale
normalized_gray = (0.299 * r_normalized + 0.587 * g_normalized + 0.114 * b_normalized)
normalized_gray = np.nan_to_num(normalized_gray, nan=0.0, posinf=1.0, neginf=0.0) # Make sure all values are in range
normalized_gray = np.clip(normalized_gray, 0, 1)
normalized_image = (normalized_gray * 255).astype(np.uint8)

# Gamma correction
gamma = 2
gamma_corrected = np.power(normalized_image / 255.0, gamma)
gamma_image = (gamma_corrected * 255).astype(np.uint8)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(gray, cmap="gray")
ax[0].set_title("Grayscale Image")
ax[0].axis("off")

ax[1].imshow(normalized_image, cmap="gray")
ax[1].set_title("Normalized Grayscale Image")
ax[1].axis("off")

ax[2].imshow(gamma_image, cmap="gray")
ax[2].set_title("Gamma Corrected Image")
ax[2].axis("off")

plt.show()

"""
grey_image_pil = Image.fromarray(gray)
grey_image_pil.save('contrast_images/grayscale/11_normal_fundus_grayscale.png')

normalized_image_pil = Image.fromarray(normalized_gray_display)
normalized_image_pil.save('contrast_images/normalized/11_normal_fundus_normalized.png')

gamma_corrected_image_pil = Image.fromarray(gamma_corrected_display)
gamma_corrected_image_pil.save('contrast_images/gamma/11_normal_fundus_gamma.png')

"""