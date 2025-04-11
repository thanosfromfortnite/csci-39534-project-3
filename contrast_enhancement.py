"""
    Erika Lin
    Erika.lin25@myhunter.cuny.edu
    CSCI 39534 Project 3 - Retinal Fundus Images
    Person 2 - Contrast Enhancement Pipeline
    Resources: Images - Kaggle
               https://www.kaggle.com/datasets/kssanjaynithish03/retinal-fundus-images/data

    t_optimum (image 1 - 11): 103, 183, 112, 97, 97, 121, 112, 74, 91, 101, 115
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open('images/original/11_normal_fundus.jpeg')
image_number = '11'
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

# Add together to create single-channel grayscale
normalized_gray = (0.299 * r_normalized + 0.587 * g_normalized + 0.114 * b_normalized)
normalized_gray = np.nan_to_num(normalized_gray, nan=0.0, posinf=1.0, neginf=0.0) 
normalized_gray = np.clip(normalized_gray, 0, 1)
normalized_image = (normalized_gray * 255).astype(np.uint8)

# Normalize pixel values [0, 1]
gray_float = gray.astype(np.float32) / 255.0
normalized_image_float = normalized_image.astype(np.float32) / 255.0

# Compute difference
diff = np.abs(gray_float - normalized_image_float)

# Gamma correction 
gamma = 2
gamma_diff = np.power(diff, gamma)

# Alpha blending
alpha = 0.5
blended = alpha * gamma_diff + (1 - alpha) * gray_float
blended_image = np.clip(blended * 255, 0, 255).astype(np.uint8)

# t_optimum from student 1
if image_number == "01":
    t_optimum = 103
elif image_number == "02":
    t_optimum = 183
elif image_number == "03":
    t_optimum = 112
elif image_number == "04":
    t_optimum = 97
elif image_number == "05":
    t_optimum = 97
elif image_number == "06":
    t_optimum = 121
elif image_number == "07":
    t_optimum = 112
elif image_number == "08":
    t_optimum = 74
elif image_number == "09":
    t_optimum = 91
elif image_number == "10":
    t_optimum = 101
elif image_number == "11":
    t_optimum = 115

I_min = np.min(blended_image)
I_max = np.max(blended_image)
piecewise_image = blended_image.copy()

# piecewise function
for i in range(blended_image.shape[0]):
    for j in range(blended_image.shape[1]):
        pixel = blended_image[i, j]
        if pixel <= t_optimum:
            piecewise_image[i, j] = int((pixel - I_min) * ((t_optimum - I_min) / (t_optimum - I_min)) + I_min)
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
ax[3].set_title("piecewise_image")
ax[3].axis("off")

plt.tight_layout()
plt.show()

"""
grey_image_pil = Image.fromarray(gray)
grey_image_pil.save('contrast_images/grayscale/11_normal_fundus_grayscale.png')

normalized_image_pil = Image.fromarray(normalized_gray_display)
normalized_image_pil.save('contrast_images/normalized/11_normal_fundus_normalized.png')

alpha_blend_image_pil = Image.fromarray(blended_image)
alpha_blend_image_pil.save('contrast_images/alpha_blend/11_normal_fundus_alpha.png')

piecewise_image_pil = Image.fromarray(piecewise_image)
piecewise_image_pil.save('contrast_images/piecewise/11_normal_fundus_piecewise.png')

"""