import cv2 as cv
import numpy as np
import torch


# torch.backends.cudnn.enabled = False
def init_addterm(height, width):
    n = torch.FloatTensor(list(range(width)))
    horizontal_term = n.expand((1, 1, height, width))  # 第一个1是batch size
    n = torch.FloatTensor(list(range(height)))
    vertical_term = n.expand((1, 1, width, height)).permute(0, 1, 3, 2)
    addterm = torch.cat((horizontal_term, vertical_term), dim=1)
    return addterm


def warp(frame, flow):
    b, c, h, w = frame.size()
    addterm = init_addterm(h, w)

    flow = flow + addterm

    horizontal_flow = torch.zeros(b, 1, frame.size(2), frame.size(3))
    vertical_flow = torch.zeros(b, 1, frame.size(2), frame.size(3))

    for i in range(b):
        horizontal_flow[i, :, :, :] = flow[i, 0, :, :].expand(1, 1, h, w)
        vertical_flow[i, :, :, :] = flow[i, 1, :, :].expand(1, 1, h, w)

    horizontal_flow = horizontal_flow * 2 / (w - 1) - 1
    vertical_flow = vertical_flow * 2 / (h - 1) - 1
    flow = torch.cat((horizontal_flow, vertical_flow), dim=1)
    flow = flow.permute(0, 2, 3, 1)
    reference_frame = torch.nn.functional.grid_sample(frame, flow)
    return reference_frame


def DIS_flow(prvs, next, color):
    prvs = np.uint8(prvs * 255)
    prvs = np.transpose(prvs, (1, 2, 0))
    prvs = cv.cvtColor(prvs, cv.COLOR_RGB2GRAY) if color else prvs[..., 0]
    next = np.uint8(next * 255)
    next = np.transpose(next, (1, 2, 0))
    next = cv.cvtColor(next, cv.COLOR_RGB2GRAY) if color else next[..., 0]
    DIS = cv.DISOpticalFlow_create()
    # DIS = cv.optflow.createOptFlow_DIS()
    Finest = 2
    DIS.setFinestScale(Finest)
    flow = DIS.calc(prvs, next, None, )
    flow = np.transpose(flow, (2, 0, 1))
    flow = torch.Tensor(np.expand_dims(flow, 0))
    return flow


def dis(frames, color):
    # for i in range(7):
    #     x = frames[0, i, :, :, :].cpu().detach().numpy()
    #     x = x.transpose(1, 2, 0)
    #     plt.imsave('./test%d.png' % (i + 1), (x))
    N = frames.size(1)
    b = frames.size(0)
    warpframes = torch.empty(frames.size(0), frames.size(1), frames.size(2), frames.size(3), frames.size(4))
    for num in range(b):
        ave_ref = frames[num, 3, :, :, :]
        for i in range(7):
            ave_frame = frames[num, i, :, :, :]

            flow = DIS_flow(ave_ref, ave_frame, color)
            warpframes[num, i, :, :, :] = warp(ave_frame.unsqueeze(0), flow)

    warpframes[:, N // 2, :, :, :] = frames[:, N // 2, :, :, :]
    return warpframes
