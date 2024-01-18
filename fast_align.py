import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from dis_flow import dis
import torch
import datetime

clean_folder = '000'
noised_folder = 'noised000var100'
output_folder = '000_wyc_var100'

# origin_dir ='E:/datasets/vimeo_septuplet/sequences_x4_lr_v2/'
# save_dir = 'E:/datasets/vimeo_septuplet/sequences_x4_lr_v2_warp/'
# train_list = 'I:/Datasets/vimeo_septuplet/testfile/2_90k_89/train_list.txt'

p_size = 15
PAD = True

clean_path = os.path.join(clean_folder, '00000000.png')
clean_frame = cv2.imread(clean_path)

pre = cur = datetime.datetime.now()


def main():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    num_images = 100
    for i in range(num_images):
        framey = []
        filename = f'{i:08d}.png'
        noised_path = os.path.join(noised_folder, filename)
        current_frame = cv2.imread(noised_path)
        framey.append(clean_frame)
        framey.append(current_frame)

        framey = np.array(framey)
        if PAD:
            framey = np.lib.pad(framey, ((0, 0), (p_size, p_size), (p_size, p_size), (0, 0)), 'constant')

        framey = np.transpose(framey, (0, 3, 1, 2))
        framey = torch.from_numpy(framey).unsqueeze(0).cuda()

        warpframe = dis(framey)
        warpframe = warpframe.cpu().detach().numpy()
        output_filename = f"{output_folder}/{i:08d}.png"
        for index2 in process_index:
            img = warpframe[0, index2, :, :, :]
            img = np.transpose(img, (1, 2, 0))
            if PAD:
                img = img[p_size:-p_size, p_size:-p_size]
            cv2.imwrite(output_filename, img)
        cur = datetime.datetime.now()
        processing_time = (cur - pre).seconds
        print('%.2fseconds, path %s finished!' % (processing_time, path_code))


if __name__ == "__main__":
    main()
