import os
import io
import requests
import zipfile
import torch
import torchvision
import numpy as np
import align
import rawpy
import imageio
from glob import glob
import matplotlib.pyplot as plt
import cv2

image_dir = './000/000var100'
output_folder = '000_hdr+_var100'
os.makedirs(output_folder, exist_ok=True)

image_paths = sorted(glob(f'{image_dir}/*.*'))
num_images = len(image_paths)
for frame_number in range(0, num_images - 2):
    images = []

    for i in range(frame_number, frame_number + 3):
        image = torchvision.io.read_image(image_paths[i])
        image = image.float() / 255
        images.append(image)

    # store the pixels in a tensor
    images = torch.stack(images)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    merged_image = align.align_and_merge(images, device=device)
    merged_image = np.transpose(merged_image.numpy(), (1, 2, 0))
    merged_image = cv2.cvtColor(merged_image, cv2.COLOR_RGB2BGR)
    merged_image = merged_image*255
    filename = f'{frame_number:08d}.png'
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, merged_image)
