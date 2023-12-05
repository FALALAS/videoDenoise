import cv2
import os

# 图片文件夹路径
image_folder = '000'  # 图片文件夹名
video_name = 'output_video000.avi'

# 视频的帧率
fps = 30

# 获取文件夹中的第一个图像文件，以确定视频的分辨率
first_image = cv2.imread(os.path.join(image_folder, '00000000.png'))

# 如果未能读取第一张图片，打印错误信息并退出
if first_image is None:
    print("无法读取第一张图片。请检查图片文件夹和图片文件。")
    exit()

# 定义视频编码器和创建 VideoWriter 对象
height, width, layers = first_image.shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

# 读取图片并添加到视频
for i in range(100):
    img_name = f"{i:08d}.png"  # 生成图片文件名
    img_path = os.path.join(image_folder, img_name)
    img = cv2.imread(img_path)

    if img is not None:
        video.write(img)

# 释放资源
video.release()
cv2.destroyAllWindows()
