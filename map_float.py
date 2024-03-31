import cv2
import numpy as np
import time
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def add_gaussian_noise(image, sigma):
    """
    向图像添加高斯噪声

    :param image: 原始图像
    :return: 添加噪声后的图像
    """
    row, col, ch = image.shape
    mean = 0

    gauss = sigma * np.random.normal(mean, 1, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss

    return noisy


def padding(frame, padding_width):
    padding_frame = cv2.copyMakeBorder(frame, padding_width, padding_width, padding_width, padding_width,
                                       cv2.BORDER_REFLECT)
    return padding_frame


start_time = time.time()

# 文件夹路径
clean_folder = './000/clean'
output_folder = '0001clean_padding5_floatvar100_sv'
os.makedirs(output_folder, exist_ok=True)

# 第一帧是干净的
clean_path = os.path.join(clean_folder, '00000000.png')
prev_frame = cv2.imread(clean_path)
prev_denoised_frame = prev_frame.astype(np.float64)
denoised_frame = prev_frame
output_path = os.path.join(output_folder, '00000000.png')
cv2.imwrite(output_path, denoised_frame)

# 参数
num_images = 100
win_size = 5
win_area = win_size * win_size
varn = 100
padding_width = win_size // 2

h = prev_frame.shape[0]
w = prev_frame.shape[1]

padding_h = h + 2 * padding_width
padding_w = w + 2 * padding_width

# 遍历图片文件
for frame_number in range(1, num_images):
    # 构造文件名
    filename = f'{frame_number:08d}.png'
    clean_path = os.path.join(clean_folder, filename)
    current_frame = cv2.imread(clean_path)
    current_frame = np.float64(current_frame)
    current_frame = add_gaussian_noise(current_frame, varn ** 0.5)
    count = 0

    current_frame = padding(current_frame, padding_width)
    prev_denoised_frame = padding(prev_denoised_frame, padding_width)
    denoised_frame = np.zeros((padding_h, padding_w, prev_frame.shape[2]), dtype=np.float64)
    for x in range(0, padding_h - win_size + 1):
        center_x = x + padding_width
        for y in range(0, padding_w - win_size + 1):
            center_y = y + padding_width
            for c in range(0, prev_frame.shape[2]):
                current_window = current_frame[x: x + win_size, y: y + win_size, c]
                diff = current_window - prev_denoised_frame[center_x, center_y, c]
                varx = np.mean(diff ** 2) - varn
                if varx < 0:
                    count = count + 1
                    varx = 0
                lam = varn / (varx + 0.1)

                factor1 = current_frame[center_x, center_y, c] / (1 + lam)
                factor2 = prev_denoised_frame[center_x, center_y, c] * lam / (1 + lam)
                denoised_frame[center_x, center_y, c] = factor1 + factor2

    denoised_frame = denoised_frame[padding_width: -padding_width, padding_width: -padding_width, :]
    prev_denoised_frame = denoised_frame
    denoised_frame = np.clip(denoised_frame, 0, 255).astype(np.uint8)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, denoised_frame)

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
