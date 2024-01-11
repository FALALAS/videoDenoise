import cv2
import numpy as np
import time
import os
import matlab.engine
from PIL import Image

eng = matlab.engine.start_matlab()
start_time = time.time()

# 文件夹路径
noised_folder = 'noised000var100'
output_folder = '000_vbm4d_var100'
os.makedirs(output_folder, exist_ok=True)

# 参数
num_images = 5
varn = 100
image_arrays = []
# 遍历图片文件
for i in range(0, num_images):
    # 构造文件名
    filename = f'{i:08d}.png'

    noised_path = os.path.join(noised_folder, filename)
    current_frame = np.array(cv2.imread(noised_path), dtype=np.float32)

    # 检查图片是否被成功加载
    if current_frame is None:
        print(f"无法读取图像文件 {filename}")
        continue

    image_arrays.append(current_frame)

sigma = varn**0.5  # Noise standard deviation. it should be in the same
# intensity range of the video
profile = 'lc'  # V-BM4D parameter profile
#  'lc' --> low complexity
#  'np' --> normal profile
do_wiener = 1  # Wiener filtering
#   1 --> enable Wiener filtering
#   0 --> disable Wiener filtering
sharpen = 1  # Sharpening
#   1 --> disable sharpening
#  >1 --> enable sharpening
deflicker = 1  # Deflickering
#   1 --> disable deflickering
#  <1 --> enable deflickering
verbose = 1  # 是否输出

images = np.stack(image_arrays, axis=-1)
denoised_images = eng.vbm4d(images, sigma, profile, do_wiener, sharpen, deflicker, verbose)

for i in range(denoised_images.size[3]):
    # 提取第i张图像
    image_array = denoised_images[:][:][:][ i]

    # 将numpy数组转换回PIL图像
    image = Image.fromarray(image_array.astype('uint8'))

    # 保存图像
    save_file_path = os.path.join(output_folder, f'{i:08d}.png')
    image.save(save_file_path)
