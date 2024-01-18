import cv2
import numpy as np
import time

start_time = time.time()
# 读取视频
cap1 = cv2.VideoCapture('tools/output_video000.avi')
cap = cv2.VideoCapture('output_videonoised000.avi')

# 视频编码器和视频输出
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video1noised.avi', fourcc, 30.0, (1280, 720))

ret1, prev1 = cap1.read()
ret, prev_frame = cap.read()

prev_frame=prev1

denoised_frame = prev_frame
out.write(denoised_frame)
frame_number = 0
win_size = 5
win_area = win_size * win_size
varn = 25

# 逐帧处理视频
while cap.isOpened():
    ret, current_frame = cap.read()
    if not ret:
        break

    # 应用去噪算法
    denoised_frame = np.zeros((prev_frame.shape[0], prev_frame.shape[1], prev_frame.shape[2]), dtype=np.uint8)

    for x in range(0, prev_frame.shape[0], win_size):
        for y in range(0, prev_frame.shape[1], win_size):
            for c in range(0, prev_frame.shape[2]):
                varx = np.float64(0)
                for i in range(0, win_size):
                    for j in range(0, win_size):
                        diff = np.float64(current_frame[x + i, y + j, c]) - np.float64(prev_frame[x + i, y + j, c])
                        diff = diff ** 2
                        varx += diff
                varx=varx/win_area
                lam = varn / varx
                for i in range(0, win_size):
                    for j in range(0, win_size):
                        factor1 = np.float64(current_frame[x + i, y + j, c]) / (1 + lam)
                        factor2 = np.float64(prev_frame[x + i, y + j, c]) * lam / (1 + lam)
                        denoised_frame[x + i, y + j, c] = np.clip(factor1 + factor2, 0, 255).astype(np.uint8)

    # 写入处理后的帧
    out.write(denoised_frame)
    prev_frame = denoised_frame
    # 显示帧
    cv2.imshow('frame', denoised_frame)
    current_time = time.time()  # 获取当前时间
    elapsed_time = current_time - start_time  # 计算经过的时间
    frame_number += 1
    print(f"已处理到第 {frame_number} 帧，用时 {elapsed_time:.2f} 秒")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
