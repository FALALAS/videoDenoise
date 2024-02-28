import cv2
import numpy as np
import os


def calculate_variance_per_channel_cv2(img):
    # 使用cv2读取图片（默认以彩色模式读取）
    if len(img.shape) == 3:
        # 分别计算每个颜色通道的方差
        variance_b = np.var(img[:, :, 0])
        variance_g = np.var(img[:, :, 1])
        variance_r = np.var(img[:, :, 2])
    else:
        variance_b = np.var(img)
        variance_g = np.var(img)
        variance_r = np.var(img)


    return variance_b, variance_g, variance_r


# 源文件夹和目标文件夹
source_folder = '../000'
variance_img = 0

# 处理每个图像
for i in range(100):
    img_name = f"{i:08d}.png"
    img_path = os.path.join(source_folder, img_name)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if image is not None:
        # 向图像添加高斯噪声
        variance_b, variance_g, variance_r = calculate_variance_per_channel_cv2(image)
        variance = (variance_b + variance_g + variance_r) / 3

    variance_img += variance

# 计算平均方差
variance_img /= 100

print(f"平均方差: {variance_img}")
