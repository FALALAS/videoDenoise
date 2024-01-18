import cv2
import numpy as np
import os
import time

start_time = time.time()


# 构建高斯金字塔
def build_gaussian_pyramid(frame, levels):
    pyramid = []
    pyramid.append(frame)
    for _ in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid


# 定义一个函数，使用金字塔来应用运动补偿
def estimate_global_motion(prev_frame, current_frame):
    # 实现一维投影的互相关来估计全局运动
    # 返回全局运动矢量
    # 计算一维投影
    proj_prev = np.sum(prev_frame, axis=0)
    proj_current = np.sum(current_frame, axis=0)

    # 计算互相关
    correlation = np.correlate(proj_prev, proj_current, "full")

    # 找到最大互相关的索引，估计运动
    displacement = np.argmax(correlation) - len(proj_prev) + 1

    return displacement


def iterative_inverse_lucas_kanade(prev_frame, current_frame, grid_step=16, iterations=3):
    # 确保图像是灰度的
    if len(prev_frame.shape) == 3:
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    if len(current_frame.shape) == 3:
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # 初始化运动矢量
    motion = np.zeros_like(prev_frame, dtype=np.float32)

    # 迭代求解
    for _ in range(iterations):
        for y in range(0, prev_frame.shape[0], grid_step):
            for x in range(0, prev_frame.shape[1], grid_step):
                # 定义窗口
                x1, x2 = max(x - 5, 0), min(x + 6, current_frame.shape[1])
                y1, y2 = max(y - 5, 0), min(y + 6, current_frame.shape[0])

                # 提取窗口
                Ix_window = cv2.Sobel(prev_frame[y1:y2, x1:x2], cv2.CV_64F, 1, 0, ksize=5)
                Iy_window = cv2.Sobel(prev_frame[y1:y2, x1:x2], cv2.CV_64F, 0, 1, ksize=5)
                It_window = current_frame[y1:y2, x1:x2].astype(np.float32) - prev_frame[y1:y2, x1:x2].astype(np.float32)

                # 求解光流方程
                A = np.vstack((Ix_window.ravel(), Iy_window.ravel())).T
                b = -It_window.ravel()
                nu = np.linalg.lstsq(A, b, rcond=None)[0]

                motion[y, x] = nu

    return motion


def apply_motion(frame, motion):
    # 根据计算出的运动矢量调整图像
    h, w = frame.shape
    flow_map = np.column_stack((np.meshgrid(np.arange(w), np.arange(h))))
    new_coords = flow_map + motion
    remapped_frame = cv2.remap(frame, new_coords, None, cv2.INTER_LINEAR)
    return remapped_frame


def align(current_frame, prev_frame, levels=3):
    if prev_frame is None:
        return build_gaussian_pyramid(current_frame, levels)

    pyramid_current = build_gaussian_pyramid(current_frame, levels)
    pyramid_prev = build_gaussian_pyramid(prev_frame, levels)

    global_motion = estimate_global_motion(pyramid_prev[-1], pyramid_current[-1])
    motion_vectors = [global_motion]

    for i in range(levels - 2, -1, -1):
        motion = iterative_inverse_lucas_kanade(pyramid_prev[i], pyramid_current[i])
        motion_vectors.append(motion)

    # 应用运动矢量
    for i, motion in enumerate(motion_vectors):
        pyramid_current[i] = apply_motion(pyramid_current[i], motion)

    return pyramid_current


def build_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []

    for i in range(len(gaussian_pyramid) - 1):
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        upsampled = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i], upsampled)
        laplacian_pyramid.append(laplacian)

    # 添加最底层的高斯层
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid


def laplacian_pyramid_fusion(pyramid1, pyramid2):
    fused_pyramid = []

    # 融合每一层的拉普拉斯金字塔
    for p1, p2 in zip(pyramid1, pyramid2):
        fused = cv2.addWeighted(p1, 0.5, p2, 0.5, 0)
        fused_pyramid.append(fused)

    # 重建图像
    fused_frame = fused_pyramid[-1]
    for i in range(len(fused_pyramid) - 1, 0, -1):
        fused_frame = cv2.pyrUp(fused_frame)
        fused_frame = cv2.add(fused_frame, fused_pyramid[i - 1])

    return fused_frame


clean_folder = "000"
noised_folder = "noised000var100"  # 指定文件夹路径
output_folder = "000_rtvd_var100"
os.makedirs(output_folder, exist_ok=True)

prev_frame = None
prev_output_pyramid = [None, None, None]
prev_channels = [None, None, None]
levels = 3  # 根据需要调整层数

num_images = 100  # 假设有 100 帧

for i in range(num_images):
    filename = f'{i:08d}.png'
    noised_path = os.path.join(noised_folder, filename)
    if i == 0:
        noised_path = os.path.join(clean_folder, filename)
    current_frame = cv2.imread(noised_path)
    # 检查图片是否被成功加载
    if current_frame is None:
        print(f"无法读取图像文件 {filename}")
        continue

    # 分离通道
    channels = cv2.split(current_frame)
    fused_channels = []
    for channel_idx, channel in enumerate(channels):

        aligned_pyramid = align(channel, prev_channels[channel_idx], levels)
        if prev_output_pyramid[channel_idx] is not None:
            aligned_laplacian_pyramid = build_laplacian_pyramid(aligned_pyramid)
            prev_laplacian_pyramid = build_laplacian_pyramid(prev_output_pyramid[channel_idx])
            fused_channel = laplacian_pyramid_fusion(aligned_laplacian_pyramid, prev_laplacian_pyramid)
        else:
            fused_channel = channel

        prev_channels[channel_idx] = channel
        prev_output_pyramid[channel_idx] = aligned_pyramid
        fused_channels.append(fused_channel)

    fused_frame = cv2.merge(fused_channels)
    # 保存处理后的帧
    output_filename = f"{output_folder}/{i:08d}.png"
    cv2.imwrite(output_filename, fused_frame)
    current_time = time.time()  # 获取当前时间
    elapsed_time = current_time - start_time  # 计算经过的时间
    print(f"已处理到第 {i} 帧，用时 {elapsed_time:.2f} 秒")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
