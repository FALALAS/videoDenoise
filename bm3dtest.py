import cv2
import numpy as np
import time
import os
from bm3d import bm3d_rgb

start_time = time.time()

# 文件夹路径
noised_folder = '001'
output_folder = '001_bm3d'
os.makedirs(output_folder, exist_ok=True)

# 参数
num_images = 100
varn = 100

# 遍历图片文件
for i in range(0, num_images):
    # 构造文件名
    filename = f'{i:08d}.png'

    noised_path = os.path.join(noised_folder, filename)
    current_frame = np.array(cv2.imread(noised_path), dtype=np.float64)

    # 检查图片是否被成功加载
    if current_frame is None:
        print(f"无法读取图像文件 {filename}")
        continue

    # 应用去噪算法
    denoised_frame = bm3d_rgb(current_frame, varn ** 0.5)
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
