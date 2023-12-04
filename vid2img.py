import cv2

# 视频文件路径
video_path = 'output_video1noised.avi'

# 读取视频
cap = cv2.VideoCapture(video_path)

# 检查视频是否打开成功
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频的总帧数
print(f"总帧数: {frame_count}")

# 对于每一帧，读取并保存为图像
for i in range(frame_count):
    ret, frame = cap.read()

    # 检查是否成功读取帧
    if not ret:
        print(f"帧 {i} 无法读取")
        break

    # 保存帧为图像
    frame_filename = f"frame_{i:08d}.png"  # 格式化文件名
    cv2.imwrite(frame_filename, frame)
    print(f"帧 {i} 已保存为 {frame_filename}")

# 释放视频对象
cap.release()
