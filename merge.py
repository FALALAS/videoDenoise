import os
import time
from skimage.restoration import denoise_invariant, denoise_tv_chambolle, denoise_bilateral
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

start_time = time.time()

# 文件夹路径
clean_folder = '001'
noised_folder = '001var625'
output_folder = '001_merge'
os.makedirs(output_folder, exist_ok=True)

# 参数
num_images = 135
varn = 625
num_merged = 10

# 遍历图片文件
for i in range(0, num_images, num_merged):
    # 构造文件名
    images = []
    for j in range(i, i + num_merged):
        filename = f'{j:08d}.png'
        noised_path = os.path.join(noised_folder, filename)
        current_frame = cv2.imread(noised_path)
        images.append(current_frame)
    images_float = [img.astype(np.float64) for img in images]
    merged_frame = np.mean(images_float, axis=0)
    merged_frame = np.clip(merged_frame, 0, 255).astype(np.uint8)

    denoised_frame = merged_frame

    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, denoised_frame)
    # 显示帧
    cv2.imshow('frame', denoised_frame)
    current_time = time.time()  # 获取当前时间
    elapsed_time = current_time - start_time  # 计算经过的时间

    print(f"已处理到第 {i} 帧，用时 {elapsed_time:.2f} 秒")
