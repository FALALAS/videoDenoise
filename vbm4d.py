import numpy as np
from scipy.fftpack import dct, idct
from scipy.fft import dst, idst
from scipy.linalg import hadamard
import pywt


def get_transform_matrix(N, transform_type, Nden=0):
    if N == 1:
        TM = np.array([[1]], dtype=float)
    elif transform_type.lower() == 'eye':
        TM = np.eye(N)
    elif transform_type.lower() == 'dct':
        TM = dct(np.eye(N), norm='ortho')
    elif transform_type.lower() == 'dst':
        TM = dst(np.eye(N), type=2, norm='ortho')
    elif transform_type.lower() == 'hadamard':
        TM = hadamard(N)
    else:
        wavelet = pywt.Wavelet(transform_type)
        LO_D, HI_D, LO_R, HI_R = wavelet.filter_bank

        # 构建变换矩阵
        TM = np.zeros((N, N))
        for i in range(N):
            # 构造一个具有单个非零元素的向量
            vec = np.zeros(N)
            vec[i] = 1

            # 使用小波重构来构建变换矩阵的每一行
            TM[i, :] = pywt.waverec([vec if j == Nden else np.zeros(len(vec)) for j in range(Nden + 1)], wavelet)

    # 标准化基元
    TM = (TM.T / np.sqrt(np.sum(TM ** 2, axis=1))).T
    # 计算逆变换矩阵
    ITM = np.linalg.inv(TM)

    return TM, ITM


def vbm4d(z, sigma=-1, profile='np', do_wiener=True, sharpen=1, deflicker=1, verbose=True):
    y_est = []

    z = z.astype(np.float32)

    # 计算最大值和最小值
    maxz = np.max(z)
    minz = np.min(z)

    # 定义缩放和平移参数
    scale = 0.7
    shift = (1 - scale) / 2

    # 对 z 进行缩放和平移
    z = (z - minz) / (maxz - minz)
    z = z * scale + shift

    # 如果 sigma 不等于 -1，则对 sigma 进行缩放
    if sigma != -1:
        sigma = sigma / (maxz - minz) * scale

    # 定义变换名称
    transform_2D_HT_name = 'bior1.5'  # 2-D spatial transform (Hard thresholding)
    transform_3rd_dim_HT_name = 'dct'  # 1-D temporal transform
    transform_4th_dim_HT_name = 'haar'  # 1-D nonlocal transform
    transform_2D_Wie_name = 'dct'  # 2-D spatial transform (Wiener filtering)
    transform_3rd_dim_Wie_name = 'dct'  # 1-D temporal transform
    transform_4th_dim_Wie_name = 'haar'  # 1-D nonlocal transform

    # 运动估计参数
    motion_est_type = 1  # 0 表示全搜索，1 表示快速搜索

    # 硬阈值处理（HT）参数
    N = 8  # 块大小
    h_minus = 4  # 向后时间范围
    h_plus = 4  # 向前时间范围
    Nstep = 4  # 每个处理体积之间的步长
    Nme = 11  # 运动估计搜索窗口
    tau_traj = -1  # 运动估计中块匹配的相似性阈值
    # 如果为 -1，则 tau_traj 将是 sigma 的函数
    M = 16  # 4-D 组的最大尺寸
    Nnl = 13  # 非局部体积搜索窗口
    tau_match = 0.5  # 非局部分组中体积匹配的相似性阈值
    lambda_thr4D = 2.7  # 硬阈值 lambda 参数
    alphaDC = sharpen  # 4-D DC 超平面的锐化参数
    alphaAC = deflicker  # 4-D AC 超平面的锐化参数
    beta = 2  # Kaiser 窗参数

    # 维纳滤波参数
    N_wiener = 8  # 块大小
    h_minus_wiener = 4  # 向后时间范围
    h_plus_wiener = 4  # 向前时间范围
    Nstep_wiener = 4  # 每个处理体积之间的步长
    Nme_wiener = 11  # 运动估计搜索窗口
    tau_traj_wiener = 0.05  # 运动估计中块匹配的相似性阈值
    M_wiener = 16  # 4-D 组的最大尺寸
    Nnl_wiener = 11  # 非局部体积搜索窗口
    tau_match_wiener = 0.5  # 非局部分组中体积匹配的相似性阈值
    beta_wiener = 1.3  # Kaiser 窗参数

    # 假设前面已经定义了 profile 及其他相关参数

    if profile == 'lc':
        N = 8
        h_minus = 3
        h_plus = 3
        Nstep = 6
        M = 1
        N_wiener = 8
        h_minus_wiener = h_minus
        h_plus_wiener = h_plus
        Nstep_wiener = Nstep
        M_wiener = M
        motion_est_type = 1
    elif profile == 'mp':
        h_minus = 7
        h_plus = 7
        Nstep = 4
        M = 32
        Nnl = 15
        h_minus_wiener = h_minus
        h_plus_wiener = h_plus
        Nstep_wiener = Nstep
        M_wiener = N
        Nnl_wiener = Nnl
        motion_est_type = 0

    # 定义变换矩阵
    H = h_plus + h_minus + 1
    Tfor, Tinv = get_transform_matrix(N, transform_2D_HT_name)
    Tfor = Tfor.astype(np.float32)
    Tinv = Tinv.astype(np.float32)

    Tfor3 = [None] * H
    Tinv3 = [None] * H
    for i in range(H):
        Tfor3[i], Tinv3[i] = get_transform_matrix(i, transform_3rd_dim_HT_name)
        Tfor3[i] = Tfor3[i].astype(np.float32)
        Tinv3[i] = Tinv3[i].astype(np.float32)

    Tfor4 = [None] * M
    Tinv4 = [None] * M
    for i in (2 ** np.arange(np.log2(M) + 1)).astype(int):
        Tfor4[i], Tinv4[i] = get_transform_matrix(i, transform_4th_dim_HT_name)
        Tfor4[i] = Tfor4[i].astype(np.float32)
        Tinv4[i] = Tinv4[i].astype(np.float32)

    # Wiener 变换矩阵
    H_wiener = h_plus_wiener + h_minus_wiener + 1
    TforW, TinvW = get_transform_matrix(N_wiener, transform_2D_Wie_name)
    TforW = TforW.astype(np.float32)
    TinvW = TinvW.astype(np.float32)

    Tfor3W = [None] * H_wiener
    Tinv3W = [None] * H_wiener
    for i in range(H_wiener):
        Tfor3W[i], Tinv3W[i] = get_transform_matrix(i, transform_3rd_dim_Wie_name)
        Tfor3W[i] = Tfor3W[i].astype(np.float32)
        Tinv3W[i] = Tinv3W[i].astype(np.float32)

    Tfor4W = [None] * M_wiener
    Tinv4W = [None] * M_wiener
    for i in (2 ** np.arange(np.log2(M_wiener) + 1)).astype(int):
        Tfor4W[i], Tinv4W[i] = get_transform_matrix(i, transform_4th_dim_Wie_name)
        Tfor4W[i] = Tfor4W[i].astype(np.float32)
        Tinv4W[i] = Tinv4W[i].astype(np.float32)

    # 创建一维 Kaiser 窗口
    window_1d = np.kaiser(N, beta)
    # 通过外积操作创建二维窗口矩阵
    Kwin = np.outer(window_1d, window_1d).astype(np.float32)
    # 创建一维 Kaiser 窗口
    window_1d = np.kaiser(N_wiener, beta_wiener)
    # 通过外积操作创建二维窗口矩阵
    Kwin_wiener = np.outer(window_1d, window_1d).astype(np.float32)


    return y_est
