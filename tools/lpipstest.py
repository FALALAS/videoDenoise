import os

import lpips
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# 加载LPIPS模型
lpips_model = lpips.LPIPS(net='alex')  # 你可以选择不同的模型，例如 'alex'

# 图像转换
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def calculate_lpips(img1_path, img2_path):
    # 加载和转换图像
    img1 = transform(Image.open(img1_path)).unsqueeze(0)
    img2 = transform(Image.open(img2_path)).unsqueeze(0)

    # 计算LPIPS
    with torch.no_grad():
        lpips_distance = lpips_model.forward(img1, img2)

    return lpips_distance.item()


# Directories of the image folders
folder1 = '../000/clean'  # Replace with the path to your '000' folder
# folder2 = '000_bm3d_var100_est'
# folder2 = '../noised000var625'
# folder2 = '../0001clean_img_var100'
# folder2 = '../0001clean_add_var625'
# folder2 = '../000/000var625'
#folder2 = '../bilateralFilter/000_var625'
folder2 = '../0001clean_paddinggray5_var625'
# folder2 = '0001clean_rtvdLiao_var100'
# folder2 = '../000_cv2_var625_bilateralFilter'

# List of PSNR values
lpips_values = []

# Loop through the image filenames
for i in range(1, 100):
    filename = f'{i:08d}.png'  # Format the filename (e.g., 0000000.png)

    # Load the corresponding images from both folders
    img1 = os.path.join(folder1, filename)
    img2 = os.path.join(folder2, filename)

    # Calculate PSNR
    lpips_score = calculate_lpips(img1, img2)
    lpips_values.append(lpips_score)

# Calculate the average PSNR
average_lpips = np.mean(lpips_values)
print(f'Average lpips: {average_lpips}')
