import cv2
import numpy as np
import time
import os

start_time = time.time()

# 文件夹路径
clean_folder = '000'
noised_folder = 'noised000var625'
output_folder = '0001clean_rtvdLiao_var625'
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
win_size = 10
win_area = win_size * win_size
varn = 625
wc = 0.25
wp = 0.7561436672967864

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
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.DISOpticalFlow_create(2)
    current_flow = flow.calc(prev_frame_gray, current_frame_gray, None)

    new_coords = flow_map - current_flow
    aligned_frame = cv2.remap(prev_denoised_frame, new_coords, None, cv2.INTER_CUBIC)
    delta_frame = cv2.subtract(current_frame, aligned_frame)

    # 应用去噪算法
    denoised_frame = np.zeros((prev_frame.shape[0], prev_frame.shape[1], prev_frame.shape[2]), dtype=np.uint8)

    for x in range(0, prev_frame.shape[0]):
        for y in range(0, prev_frame.shape[1]):
            for c in range(0, prev_frame.shape[2]):
                Iwiener = np.float64(delta_frame[x, y, c]) ** 2 / (np.float64(delta_frame[x, y, c]) ** 2 + varn ** 2)
                I_delta = Iwiener * np.float64(delta_frame[x, y, c])

                factor1 = wc * np.float64(current_frame[x, y, c])
                factor2 = wp * (np.float64(aligned_frame[x, y, c]) + I_delta)
                denoised_frame[x, y, c] = np.clip(factor1 + factor2, 0, 255).astype(np.uint8)

    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, denoised_frame)
    prev_denoised_frame = cv2.bilateralFilter(current_frame, 10, 80, 80)
    prev_frame = current_frame

    current_time = time.time()  # 获取当前时间
    elapsed_time = current_time - start_time  # 计算经过的时间
    frame_number += 1
    print(f"已处理到第 {frame_number} 帧，用时 {elapsed_time:.2f} 秒")

# 释放资源

cv2.destroyAllWindows()
