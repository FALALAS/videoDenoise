import os
import time
from skimage.restoration import denoise_invariant, denoise_tv_chambolle, denoise_bilateral
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
start_time = time.time()

# 文件夹路径
clean_folder = '000'
noised_folder = 'noised000var225'
output_folder = '000_cv2_var225_bilateralFilter'
os.makedirs(output_folder, exist_ok=True)

# 参数
num_images = 100
varn = 225

# 遍历图片文件
for i in range(0, num_images):
    # 构造文件名
    filename = f'{i:08d}.png'

    noised_path = os.path.join(noised_folder, filename)
    current_frame = cv2.imread(noised_path)

    # 检查图片是否被成功加载
    if current_frame is None:
        print(f"无法读取图像文件 {filename}")
        continue

    # 应用去噪算法

    # denoised_frame = cv2.fastNlMeansDenoisingColored(current_frame, None, 13, 13, 7, 21)
    denoised_frame = cv2.bilateralFilter(current_frame, 10, 40, 40)
    denoised_frame = np.clip(denoised_frame, 0, 255).astype(np.uint8)

    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, denoised_frame)
    # 显示帧
    cv2.imshow('frame', denoised_frame)
    current_time = time.time()  # 获取当前时间
    elapsed_time = current_time - start_time  # 计算经过的时间

    print(f"已处理到第 {i} 帧，用时 {elapsed_time:.2f} 秒")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源

cv2.destroyAllWindows()
# List of PSNR values
psnr_values = []

# Loop through the image filenames
for i in range(1, 100):
    filename = f'{i:08d}.png'  # Format the filename (e.g., 0000000.png)

    # Load the corresponding images from both folders
    img1 = cv2.imread(os.path.join(output_folder, filename))
    img2 = cv2.imread(os.path.join(clean_folder, filename))

    # Calculate PSNR
    psnr = compare_psnr(img1, img2)
    psnr_values.append(psnr)

# Calculate the average PSNR
average_psnr = np.mean(psnr_values)
print(f'Average PSNR: {average_psnr}')