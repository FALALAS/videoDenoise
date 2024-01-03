import cv2
import numpy as np
import os


def add_gaussian_noise(image):
    """
    向图像添加高斯噪声

    :param image: 原始图像
    :return: 添加噪声后的图像
    """
    row, col, ch = image.shape
    mean = 0
    var = 200
    sigma = var ** 0.5

    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss

    return np.clip(noisy, 0, 255).astype(np.uint8)


# 源文件夹和目标文件夹
source_folder = '000'
target_folder = 'noised000var200'

# 如果目标文件夹不存在，则创建它
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 处理每个图像
for i in range(100):
    img_name = f"{i:08d}.png"
    img_path = os.path.join(source_folder, img_name)
    image = cv2.imread(img_path)

    if image is not None:
        # 向图像添加高斯噪声
        noised_image = add_gaussian_noise(image)

        # 保存处理后的图像
        cv2.imwrite(os.path.join(target_folder, img_name), noised_image)
