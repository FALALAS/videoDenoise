import cv2
import numpy as np
import time
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

start_time = time.time()

# 文件夹路径
clean_folder = './000/clean'
noised_folder = './000/000var625'
output_folder = '0001clean_rtevd_var625'
exp_folder = "exp"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(exp_folder, exist_ok=True)


# 第一帧的双边滤波
noised_path = os.path.join(noised_folder, '00000000.png')
prev_frame = cv2.imread(noised_path)
prev_denoised_frame = cv2.bilateralFilter(prev_frame, 5, 80, 80)

'''
clean_path = os.path.join(clean_folder, '00000000.png')
prev_frame = cv2.imread(clean_path)
prev_denoised_frame = prev_frame
'''
# 第一帧是干净的

denoised_frame = prev_denoised_frame
output_path = os.path.join(output_folder, '00000000.png')
cv2.imwrite(output_path, denoised_frame)

# 参数
num_images = 100
win_size = 5
win_area = win_size * win_size
varn = 625
padding_width = win_size // 2

sigma_i = 41
sigma_d = 39
sigma_t = 38

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

    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2YUV)[:, :, 0]
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2YUV)[:, :, 0]
    flow = cv2.DISOpticalFlow_create(2)
    flow.setFinestScale(0)
    current_flow = flow.calc(prev_frame_gray, current_frame_gray, None)

    '''
    flow = cv2.optflow.DenseRLOFOpticalFlow_create()
    current_flow = flow.calc(prev_denoised_frame, current_frame, None)
    '''

    new_coords = flow_map - current_flow
    aligned_frame = cv2.remap(prev_denoised_frame, new_coords, None, cv2.INTER_CUBIC)

    # current_frame_gray = cv2.copyMakeBorder(current_frame_gray, padding_width, padding_width, padding_width,
    #                                         padding_width,
    #                                         cv2.BORDER_CONSTANT, value=0)
    # aligned_frame_gray = cv2.copyMakeBorder(aligned_frame_gray, padding_width, padding_width, padding_width,
    #                                         padding_width,
    #                                         cv2.BORDER_CONSTANT, value=0)
    # current_frame = cv2.copyMakeBorder(current_frame, padding_width, padding_width, padding_width, padding_width,
    #                                    cv2.BORDER_CONSTANT, value=0)
    # aligned_frame = cv2.copyMakeBorder(aligned_frame, padding_width, padding_width, padding_width, padding_width,
    #                                    cv2.BORDER_CONSTANT, value=0)

    # 应用去噪算法
    current_frame = current_frame.astype(np.float32)
    aligned_frame = aligned_frame.astype(np.float32)
    exp_Ip_I = np.exp(-(aligned_frame - current_frame) ** 2. / 2 / sigma_t ** 2)
    cv2.imwrite(os.path.join(exp_folder, filename), exp_Ip_I[:, :, 0] * 255)
    It = aligned_frame * exp_Ip_I + current_frame * (1 - exp_Ip_I)
    denoised_frame = (cv2.bilateralFilter(It, d=5, sigmaSpace=sigma_d, sigmaColor=sigma_i)).clip(0, 255).astype(np.uint8)

    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, denoised_frame)
    prev_denoised_frame = denoised_frame
    prev_frame = current_frame.astype(np.uint8)

    current_time = time.time()  # 获取当前时间
    elapsed_time = current_time - start_time  # 计算经过的时间
    print(f"已处理到第 {frame_number} 帧，用时 {elapsed_time:.2f} 秒异常窗口 NULL 个")

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
