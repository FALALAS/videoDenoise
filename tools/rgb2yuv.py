import cv2
import os

# 图片所在的文件夹路径
folder_path = '../000'
output_folder_path = '../000_yuv'
os.makedirs(output_folder_path, exist_ok=True)
# 循环读取图片
for i in range(100):
    # 构建图片文件的路径
    image_path = os.path.join(folder_path, f'{i:08d}.png')

    # 使用OpenCV读取图片
    image = cv2.imread(image_path)

    # 检查图片是否成功读取
    if image is not None:
        # 将图片从BGR转换为YUV格式
        image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        cv2.imwrite(os.path.join(output_folder_path, f'{i:08d}.png'), image_yuv[:, :, 0])
