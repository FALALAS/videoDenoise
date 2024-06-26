import cv2
import numpy as np
import os


def add_gaussian_noise(image):
    """
    向图像添加高斯噪声

    :param image: 原始图像
    :return: 添加噪声后的图像
    """
    row, col = image.shape
    mean = 0
    var = 25
    sigma = var

    gauss = sigma * np.random.normal(mean, 1, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss

    return np.clip(noisy, 0, 255).astype(np.uint8)


# 源文件夹和目标文件夹
source_folder = '../000'
target_folder = '../noised000sigma25'

# 如果目标文件夹不存在，则创建它
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 参数
num_images = 100
win_size = 5
padding_width = win_size // 2

# 处理每个图像
for i in range(100):
    img_name = f"{i:08d}.png"
    img_path = os.path.join(source_folder, img_name)
    image = cv2.imread(img_path)

    h = image.shape[0]
    w = image.shape[1]

    # 向图像添加高斯噪声
    noised_image = np.zeros((h, w, image.shape[2]), dtype=np.uint8)
    for x in range(0, h, win_size):
        for y in range(0, w, win_size):
            for c in range(0, 3):
                current_window = image[x: x + win_size, y: y + win_size, c]
                noised_window = add_gaussian_noise(current_window)

                noised_image[x: x + win_size, y: y + win_size, c] = noised_window

    # 保存处理后的图像
    cv2.imwrite(os.path.join(target_folder, img_name), noised_image)
