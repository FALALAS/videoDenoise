import cv2
from PIL import Image
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def calculate_psnr(img1, img2):
    # Convert images to numpy arrays of float32 type
    img1 = np.array(img1, dtype=np.float32)
    img2 = np.array(img2, dtype=np.float32)

    # Compute the Mean Squared Error (MSE) between the two images
    mse = np.mean((img1 - img2) ** 2)

    # Handle the case of a zero MSE (identical images)
    if mse == 0:
        return float('inf')

    # Calculate the PSNR (Peak Signal-to-Noise Ratio)
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr


# Directories of the image folders
folder1 = '../000'  # Replace with the path to your '000' folder
# folder2 = '000_bm3d_var100_est'
# folder2 = '../noised000var625'
# folder2 = '../0001clean_img_var100'
# folder2 = '../0001clean_add_var625'
# folder2 = '../0001clean_rtvdLiao_var100'
# folder2 = '../000_bm3d_var625'
# folder2 = '0001clean_rtvdLiao_var100'
folder2 = '../000_cv2_var625_NlM'

# List of PSNR values
psnr_values = []

# Loop through the image filenames
for i in range(1, 100):
    filename = f'{i:08d}.png'  # Format the filename (e.g., 0000000.png)

    # Load the corresponding images from both folders
    img1 = cv2.imread(os.path.join(folder1, filename))
    img2 = cv2.imread(os.path.join(folder2, filename))

    # Calculate PSNR
    psnr = compare_psnr(img1, img2)
    psnr_values.append(psnr)

# Calculate the average PSNR
average_psnr = np.mean(psnr_values)
print(f'Average PSNR: {average_psnr}')
