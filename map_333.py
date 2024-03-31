import cv2
import numpy as np
import time
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def padding(frame, padding_width):
    padding_frame = cv2.copyMakeBorder(frame, padding_width, padding_width, padding_width, padding_width,
                                       cv2.BORDER_CONSTANT, value=0)
    return padding_frame


start_time = time.time()

# 文件夹路径
clean_folder = '000'
noised_folder = 'noised000var625'
output_folder = '0001clean_333_var625'
os.makedirs(output_folder, exist_ok=True)

'''
noised_path = os.path.join(noised_folder, '00000000.png')
prev_frame = cv2.imread(noised_path)
denoised_frame = cv2.bilateralFilter(prev_frame, 10, 40, 40)
'''
noised_path = os.path.join(noised_folder, '00000000.png')
prev_frame = cv2.imread(noised_path)

clean_path = os.path.join(clean_folder, '00000000.png')
denoised_frame = cv2.imread(clean_path)
prev_denoised_frame = denoised_frame
output_path = os.path.join(output_folder, '00000000.png')
cv2.imwrite(output_path, denoised_frame)

# 参数
num_images = 100
win_size = 3
win_area = win_size * win_size
varn = 625
padding_width = win_size // 2
T = 100

h = prev_frame.shape[0]
w = prev_frame.shape[1]

padding_h = h + 2 * padding_width
padding_w = w + 2 * padding_width

# 遍历图片文件
for frame_number in range(1, num_images - 1):
    # 构造文件名
    filename = f'{frame_number:08d}.png'
    noised_path = os.path.join(noised_folder, filename)
    current_frame = cv2.imread(noised_path)
    current_denoised_frame = cv2.bilateralFilter(current_frame, 10, 80, 80)

    buffer_filename = f'{frame_number + 1:08d}.png'
    buffer_path = os.path.join(noised_folder, buffer_filename)
    buffer_frame = cv2.imread(buffer_path)

    prev_frame_y = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2YUV)[:, :, 0]
    current_frame_y = cv2.cvtColor(current_frame, cv2.COLOR_BGR2YUV)[:, :, 0]
    buffer_frame_y = cv2.cvtColor(buffer_frame, cv2.COLOR_BGR2YUV)[:, :, 0]
    prev_denoised_frame_y = cv2.cvtColor(prev_denoised_frame, cv2.COLOR_BGR2YUV)[:, :, 0]

    prev_frame_y = padding(prev_frame_y, padding_width)
    current_frame_y = padding(current_frame_y, padding_width)
    buffer_frame_y = padding(buffer_frame_y, padding_width)
    prev_denoised_frame_y = padding(prev_denoised_frame_y, padding_width)

    current_frame = padding(current_frame, padding_width)
    prev_denoised_frame = padding(prev_denoised_frame, padding_width)
    current_denoised_frame = padding(current_denoised_frame, padding_width)

    # 应用去噪算法
    count = 0
    denoised_frame = np.zeros((padding_h, padding_w, prev_frame.shape[2]), dtype=np.float64)
    for x in range(0, padding_h - win_size + 1):
        center_x = x + padding_width
        for y in range(0, padding_w - win_size + 1):
            center_y = y + padding_width

            prev_window = prev_frame_y[x: x + win_size, y: y + win_size]
            current_window = current_frame_y[x: x + win_size, y: y + win_size]
            buffer_window = buffer_frame_y[x: x + win_size, y: y + win_size]

            miu = prev_denoised_frame_y[center_x, center_y].astype(np.float64)

            prev_diff = prev_window.astype(np.float64) - miu
            current_diff = current_window.astype(np.float64) - miu
            buff_diff = buffer_window.astype(np.float64) - miu

            diff = np.stack((prev_diff, current_diff, buff_diff))

            varx = np.mean(diff ** 2) - varn
            if varx < 0:
                varx = 0

            if varx > T:
                denoised_frame[center_x, center_y] = current_denoised_frame[center_x, center_y].astype(np.float64)
            else:
                count = count + 1
                lam = varn / (varx + 1e-16)
                factor1 = np.float64(current_frame[center_x, center_y]) / (1 + lam)
                factor2 = np.float64(prev_denoised_frame[center_x, center_y]) * lam / (1 + lam)
                denoised_frame[center_x, center_y] = factor1 + factor2

    denoised_frame = denoised_frame[padding_width: -padding_width, padding_width: -padding_width, :]
    denoised_frame = np.clip(denoised_frame, 0, 255).astype(np.uint8)

    # denoised_frame = cv2.bilateralFilter(denoised_frame, 10, 10, 10)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, denoised_frame)
    prev_frame = current_frame[padding_width: -padding_width, padding_width: -padding_width, :]
    prev_denoised_frame = denoised_frame

    current_time = time.time()  # 获取当前时间
    elapsed_time = current_time - start_time  # 计算经过的时间
    print(f"已处理到第 {frame_number} 帧，用时 {elapsed_time:.2f} 秒，异常窗口 {count} 个")

# List of PSNR values
psnr_values = []

# Loop through the image filenames
for i in range(1, 100):
    filename = f'{i:08d}.png'  # Format the filename (e.g., 0000000.png)

    # Load the corresponding images from both folders
    img1 = cv2.imread(os.path.join(output_folder, filename))
    img2 = cv2.imread(os.path.join(clean_folder, filename))

    # Calculate PSNR
    psnr = compare_psnr(img1, img2)
    psnr_values.append(psnr)

# Calculate the average PSNR
average_psnr = np.mean(psnr_values)
print(f'Average PSNR: {average_psnr}')
