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
def align(current_frame, prev_frame, levels=3):
    # 如果前帧为空，则使用当前帧构建高斯金字塔
    if prev_frame is None:
        return build_gaussian_pyramid(current_frame, levels)

    # 构建当前帧和前帧的高斯金字塔
    pyramid_current = build_gaussian_pyramid(current_frame, levels)
    pyramid_prev = build_gaussian_pyramid(prev_frame, levels)

    # 从最低分辨率开始估计光流
    flow = cv2.DISOpticalFlow_create(0)
    flow.setFinestScale(0)
    flow.setPatchSize(40)
    current_level_flow = flow.calc(pyramid_prev[-1], pyramid_current[-1], None)


    # current_level_flow = cv2.calcOpticalFlowFarneback(pyramid_prev[-1], pyramid_current[-1], None,0.5, 1, 460800,
    # 10, 7, 1.5, 0)

    # 在每一层上应用光流
    for i in range(levels, -1, -1):
        h, w = pyramid_current[i].shape[:2]
        flow_map = np.meshgrid(np.arange(w), np.arange(h))
        flow_map = np.stack(flow_map, axis=-1).astype(np.float32)  # 调整为三维数组
        new_coords = flow_map - current_level_flow
        pyramid_current[i] = cv2.remap(pyramid_current[i], new_coords, None, cv2.INTER_LINEAR)

        # 将当前层光流上采样到下一层
        if i > 0:
            current_level_flow = cv2.pyrUp(current_level_flow)

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
