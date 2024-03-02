import cv2
import numpy as np
import time
import os

start_time = time.time()

# 文件夹路径
clean_folder = '000'
noised_folder = 'noised000var2500'
output_folder = '0001clean_padding7_var2500'
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
win_size = 7
win_area = win_size * win_size
varn = 2500
padding_width = win_size // 2

h = prev_frame.shape[0]
w = prev_frame.shape[1]
flow_map = np.meshgrid(np.arange(w), np.arange(h))
flow_map = np.stack(flow_map, axis=-1).astype(np.float32)  # 调整为三维数组

padding_h = h + 2 * padding_width
padding_w = w + 2 * padding_width

# 遍历图片文件
for frame_number in range(1, num_images):
    # 构造文件名
    filename = f'{frame_number:08d}.png'

    noised_path = os.path.join(noised_folder, filename)
    current_frame = cv2.imread(noised_path)

    # 检查图片是否被成功加载
    if current_frame is None:
        print(f"无法读取图像文件 {filename}")
        continue

    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    prev_frame_gray = cv2.cvtColor(prev_denoised_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.DISOpticalFlow_create(2)
    current_flow = flow.calc(prev_frame_gray, current_frame_gray, None)

    new_coords = flow_map - current_flow
    aligned_frame = cv2.remap(prev_denoised_frame, new_coords, None, cv2.INTER_CUBIC)

    current_frame = cv2.copyMakeBorder(current_frame, padding_width, padding_width, padding_width, padding_width,
                                       cv2.BORDER_CONSTANT, value=0)
    aligned_frame = cv2.copyMakeBorder(aligned_frame, padding_width, padding_width, padding_width, padding_width,
                                       cv2.BORDER_CONSTANT, value=0)

    # 应用去噪算法
    denoised_frame = np.zeros((padding_h, padding_w, prev_frame.shape[2]), dtype=np.uint8)
    for x in range(0, padding_h - win_size + 1):
        for y in range(0, padding_w - win_size + 1):
            for c in range(0, prev_frame.shape[2]):
                current_window = current_frame[x: x + win_size, y: y + win_size, c]
                aligned_window = aligned_frame[x: x + win_size, y: y + win_size, c]

                diff = current_window.astype(np.float64) - aligned_window.astype(np.float64)
                varx = np.mean(diff ** 2)
                varx = np.absolute(varx - varn)
                lam = varn / (varx + 1e-16)

                center_x = x + padding_width
                center_y = y + padding_width
                factor1 = np.float64(current_frame[center_x, center_y, c]) / (1 + lam)
                factor2 = np.float64(aligned_frame[center_x, center_y, c]) * lam / (1 + lam)
                denoised_frame[center_x, center_y, c] = np.clip(factor1 + factor2, 0, 255).astype(np.uint8)
    denoised_frame = denoised_frame[padding_width: -padding_width, padding_width: -padding_width, :]
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, denoised_frame)
    prev_frame = denoised_frame

    current_time = time.time()  # 获取当前时间
    elapsed_time = current_time - start_time  # 计算经过的时间
    print(f"已处理到第 {frame_number} 帧，用时 {elapsed_time:.2f} 秒")
