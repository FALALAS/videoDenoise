import cv2
import numpy as np
import time
import os

start_time = time.time()

# 文件夹路径
clean_folder = '000'
noised_folder = 'noised000var2500'
output_folder = 'gray'
os.makedirs(output_folder, exist_ok=True)

# 第一帧是干净的
clean_path = os.path.join(clean_folder, '00000000.png')
prev_frame = cv2.imread(clean_path)
prev_denoised_frame = prev_frame
denoised_frame = prev_frame
output_path = os.path.join(output_folder, '00000000.png')
cv2.imwrite(output_path, denoised_frame)

# 参数
num_images = 100
frame_number = 0
win_size = 5
win_area = win_size * win_size
varn = 2500

h = prev_frame.shape[0]
w = prev_frame.shape[1]
flow_map = np.meshgrid(np.arange(w), np.arange(h))
flow_map = np.stack(flow_map, axis=-1).astype(np.float32)  # 调整为三维数组

# 遍历图片文件
for i in range(1, num_images):
    # 构造文件名
    filename = f'{i:08d}.png'
    noised_path = os.path.join(noised_folder, filename)
    current_frame = cv2.imread(noised_path)
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, current_frame_gray)


