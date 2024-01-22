import os
import time
from skimage.restoration import denoise_invariant, denoise_tv_chambolle, denoise_bilateral
import cv2
import numpy as np

start_time = time.time()

# 文件夹路径
noised_folder = 'noised000var625'
output_folder = '000_cv2_var625_NlM'
os.makedirs(output_folder, exist_ok=True)

# 参数
num_images = 100
varn = 625

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

    denoised_frame = cv2.fastNlMeansDenoisingColored(current_frame, None, 5, 5, 7, 21)
    # denoised_frame = cv2.bilateralFilter(current_frame, 10, 80, 80)
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
