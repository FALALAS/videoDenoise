import cv2
import numpy as np
import time
import os
from bm3d import bm3d_rgb

start_time = time.time()

# 文件夹路径
clean_folder = '000'
noised_folder = 'noised000var100'
output_folder = '000_cv2_var100_multi'
os.makedirs(output_folder, exist_ok=True)

clean_path = os.path.join(clean_folder, '00000000.png')
prev_frame = cv2.imread(clean_path)
# 参数
num_images = 100
varn = 100

# 遍历图片文件
for i in range(1, num_images-1):
    # 构造文件名
    filename = f'{i + 1:08d}.png'
    noised_path = os.path.join(noised_folder, filename)
    next_frame = cv2.imread(noised_path)

    filename = f'{i:08d}.png'
    noised_path = os.path.join(noised_folder, filename)
    current_frame = cv2.imread(noised_path)

    noised_frames = [prev_frame, current_frame, next_frame]

    # 应用去噪算法
    denoised_frame = cv2.fastNlMeansDenoisingColoredMulti(noised_frames, 1, 3, None, 5, 5, 7, 21)
    denoised_frame = np.clip(denoised_frame, 0, 255).astype(np.uint8)

    prev_frame = denoised_frame

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
