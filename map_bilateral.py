import cv2
import numpy as np
import time
import os

def adjusted_decay(x):
    # 定义参数
    a = 6.56
    b = 0.3
    c = 1.05
    d = 1

    # 计算函数值
    return a * np.exp(-b * (x - c)) + d

start_time = time.time()

# 文件夹路径
noised_folder = 'noised000var625'
output_folder = '000_bi_var625'
os.makedirs(output_folder, exist_ok=True)

noised_path = os.path.join(noised_folder, '00000000.png')
prev_frame = cv2.imread(noised_path)
prev_denoised_frame = cv2.bilateralFilter(prev_frame, 10, 80, 80)
denoised_frame = prev_denoised_frame
output_path = os.path.join(output_folder, '00000000.png')
cv2.imwrite(output_path, denoised_frame)

# 参数
num_images = 100
win_size = 2
win_area = win_size * win_size
varn = 625

h = prev_frame.shape[0]
w = prev_frame.shape[1]
flow_map = np.meshgrid(np.arange(w), np.arange(h))
flow_map = np.stack(flow_map, axis=-1).astype(np.float32)  # 调整为三维数组

# 遍历图片文件
for frame_number in range(1, num_images):
    # 构造文件名
    filename = f'{frame_number:08d}.png'
    noised_path = os.path.join(noised_folder, filename)
    current_frame = cv2.imread(noised_path)

    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.DISOpticalFlow_create(2)
    flow.setFinestScale(0)
    current_flow = flow.calc(prev_frame_gray, current_frame_gray, None)

    '''
    flow = cv2.optflow.DenseRLOFOpticalFlow_create()
    current_flow = flow.calc(prev_denoised_frame, current_frame, None)
    '''

    new_coords = flow_map - current_flow
    aligned_frame = cv2.remap(prev_denoised_frame, new_coords, None, cv2.INTER_CUBIC)
    aligned_frame_gray = cv2.cvtColor(aligned_frame, cv2.COLOR_BGR2GRAY)

    # 应用去噪算法
    denoised_frame = np.zeros((prev_frame.shape[0], prev_frame.shape[1], prev_frame.shape[2]), dtype=np.uint8)
    for x in range(0, prev_frame.shape[0], win_size):
        for y in range(0, prev_frame.shape[1], win_size):
            varx = np.float64(0)
            for i in range(0, win_size):
                for j in range(0, win_size):
                    diff = np.float64(current_frame_gray[x + i, y + j]) - np.float64(aligned_frame_gray[x + i, y + j])
                    diff = diff ** 2
                    varx += diff
            varx = varx / win_area
            lam = varn / (varx + 1e-16)
            for i in range(0, win_size):
                for j in range(0, win_size):
                    factor1 = np.float64(current_frame[x + i, y + j]) / (1 + lam)
                    factor2 = np.float64(aligned_frame[x + i, y + j]) * lam / (1 + lam)
                    denoised_frame[x + i, y + j] = np.clip(factor1 + factor2, 0, 255).astype(np.uint8)

    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, denoised_frame)
    prev_denoised_frame = cv2.bilateralFilter(denoised_frame, 10, 50, 50)
    prev_frame = current_frame

    current_time = time.time()  # 获取当前时间
    elapsed_time = current_time - start_time  # 计算经过的时间
    print(f"已处理到第 {frame_number} 帧，用时 {elapsed_time:.2f} 秒")
