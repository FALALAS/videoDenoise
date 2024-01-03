import cv2
import numpy as np
import time
import os

start_time = time.time()

# 文件夹路径
clean_folder = '000'
noised_folder = 'noised000var200'
output_folder = '0001clean_img_var200'

# 第一帧是干净的
clean_path = os.path.join(clean_folder, '00000000.png')
prev_frame = cv2.imread(clean_path)
denoised_frame = prev_frame
output_path = os.path.join(output_folder, '00000000.png')
cv2.imwrite(output_path, denoised_frame)

# 参数
num_images = 100
frame_number = 0
win_size = 10
win_area = win_size * win_size
varn = 200

# 遍历图片文件
for i in range(1, num_images):
    # 构造文件名
    filename = f'{i:08d}.png'

    noised_path = os.path.join(noised_folder, filename)
    current_frame = cv2.imread(noised_path)

    # 检查图片是否被成功加载
    if current_frame is None:
        print(f"无法读取图像文件 {filename}")
        continue

    # 应用去噪算法
    denoised_frame = np.zeros((prev_frame.shape[0], prev_frame.shape[1], prev_frame.shape[2]), dtype=np.uint8)

    '''
    for x in range(0, prev_frame.shape[0], win_size):
        for y in range(0, prev_frame.shape[1], win_size):
            for c in range(0, prev_frame.shape[2]):
                varx = np.float64(0)
                diff = np.float64(0)
                for i in range(0, win_size):
                    for j in range(0, win_size):
                        yn = np.float64(current_frame[x + i, y + j, c])
                        xn_1 = np.float64(prev_frame[x + i, y + j, c])
                        diff += yn ** 2 + xn_1 ** 2 + 2 * yn * xn_1

                varx = (diff / win_area) / (varn + diff / win_area)
                lam = 1 - varx
                for i in range(0, win_size):
                    for j in range(0, win_size):
                        factor1 = np.float64(current_frame[x + i, y + j, c]) / (1 + lam)
                        factor2 = np.float64(prev_frame[x + i, y + j, c]) * lam / (1 + lam)
                        denoised_frame[x + i, y + j, c] = np.clip(factor1 + factor2, 0, 255).astype(np.uint8)
    '''

    for x in range(0, prev_frame.shape[0], win_size):
        for y in range(0, prev_frame.shape[1], win_size):
            for c in range(0, prev_frame.shape[2]):
                varx = np.float64(0)
                for i in range(0, win_size):
                    for j in range(0, win_size):
                        diff = np.float64(current_frame[x + i, y + j, c]) - np.float64(prev_frame[x + i, y + j, c])
                        diff = diff ** 2
                        varx += diff
                varx = varx / win_area
                lam = varn / varx
                for i in range(0, win_size):
                    for j in range(0, win_size):
                        factor1 = np.float64(current_frame[x + i, y + j, c]) / (1 + lam)
                        factor2 = np.float64(prev_frame[x + i, y + j, c]) * lam / (1 + lam)
                        denoised_frame[x + i, y + j, c] = np.clip(factor1 + factor2, 0, 255).astype(np.uint8)

    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, denoised_frame)
    prev_frame = denoised_frame
    # 显示帧
    cv2.imshow('frame', denoised_frame)
    current_time = time.time()  # 获取当前时间
    elapsed_time = current_time - start_time  # 计算经过的时间
    frame_number += 1
    print(f"已处理到第 {frame_number} 帧，用时 {elapsed_time:.2f} 秒")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源

cv2.destroyAllWindows()
