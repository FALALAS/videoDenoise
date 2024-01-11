import os
from glob import glob
import time
import cv2
import numpy as np
import torch
import torchvision

import align


def load_jpeg_images(image_paths):
    """loads rgb pixels from jpeg images"""
    images = []
    for path in image_paths:
        image = torchvision.io.read_image(path)
        image = image.float() / 255
        images.append(image)

    # store the pixels in a tensor
    images = torch.stack(images)

    print(f'burst of shape {list(images.shape)} loaded')
    return images


start_time = time.time()

# 文件夹路径
clean_folder = '000'
noised_folder = 'noised000var100'
output_folder = '0001clean_hdr+_var100'

# 第一帧是干净的
clean_path = os.path.join(clean_folder, '00000000.png')
prev_frame = torchvision.io.read_image(clean_path)
denoised_frame = prev_frame
output_path = os.path.join(output_folder, '00000000.png')
output_frame = np.transpose(denoised_frame.numpy(), (1, 2, 0))
cv2.imwrite(output_path, output_frame)

prev_frame = prev_frame.float() / 255

# 参数
num_images = 100
frame_number = 0
varn = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 遍历图片文件
for i in range(1, num_images):
    # 构造文件名
    filename = f'{i:08d}.png'

    noised_path = os.path.join(noised_folder, filename)
    current_frame = torchvision.io.read_image(noised_path)
    current_frame = current_frame.float() / 255

    # 检查图片是否被成功加载
    if current_frame is None:
        print(f"无法读取图像文件 {filename}")
        continue
    images = [prev_frame, current_frame]
    images = torch.stack(images)
    # 应用去噪算法
    denoised_frame = align.align_and_merge(images, device=device)
    output_frame = np.transpose(denoised_frame.numpy(), (1, 2, 0))

    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, output_frame)
    prev_frame = denoised_frame
    # 显示帧
    cv2.imshow('frame', output_frame)
    current_time = time.time()  # 获取当前时间
    elapsed_time = current_time - start_time  # 计算经过的时间
    frame_number += 1
    print(f"已处理到第 {frame_number} 帧，用时 {elapsed_time:.2f} 秒")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源

cv2.destroyAllWindows()
