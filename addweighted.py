import cv2
import numpy as np
import time
import os

start_time = time.time()

# 文件夹路径
clean_folder = '000'
noised_folder = 'noised000var625'
output_folder = '0001clean_add_var625'
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

varn = 625

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
    '''
    flow = cv2.optflow.DenseRLOFOpticalFlow_create()
    current_flow = flow.calc(prev_frame, current_frame, None)
    '''
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.DISOpticalFlow_create(2)
    current_flow = flow.calc(prev_frame_gray, current_frame_gray, None)
    new_coords = flow_map - current_flow

    aligned_frame = cv2.remap(prev_denoised_frame, new_coords, None, cv2.INTER_CUBIC)
    # 应用去噪算法
    denoised_frame = cv2.addWeighted(current_frame, 0.21, aligned_frame, 0.81, 0)

    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, denoised_frame)
    prev_denoised_frame = denoised_frame = cv2.bilateralFilter(current_frame, 10, 80, 80)
    prev_frame = current_frame

    current_time = time.time()  # 获取当前时间
    elapsed_time = current_time - start_time  # 计算经过的时间
    frame_number += 1
    print(f"已处理到第 {frame_number} 帧，用时 {elapsed_time:.2f} 秒")


