import argparse, os, cv2
import numpy as np
from tqdm import tqdm


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--lr_dir', type=str, default=r"D:\videoDenoise\noised000var2500")
    args.add_argument('--rec_dir', type=str, default=r"D:\videoDenoise\000")
    args.add_argument('--save_path', type=str, default=r"D:\videoDenoise\result.avi")
    args.add_argument('--fps', type=int, default=5)
    args.add_argument('--img_height', type=int, default=720)
    args.add_argument('--img_width', type=int, default=1280)

    return args.parse_args()


def main(args):
    lr_path = args.lr_dir
    rec_path = args.rec_dir
    lr_list = sorted(os.listdir(lr_path))
    rec_list = sorted(os.listdir(rec_path))
    num_frames = len(lr_list)
    v_pixel = args.img_width // num_frames
    direction, border = 1, 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(args.save_path, fourcc, args.fps,
                            (args.img_width, args.img_height))
    for lr_file, rec_file in tqdm(zip(lr_list, rec_list)):
        lr = cv2.imread(os.path.join(lr_path, lr_file))
        rec = cv2.imread(os.path.join(rec_path, rec_file))
        frame = np.concatenate([rec[:, 0: border, :], lr[:, border: args.img_width, :]], axis=1)
        cv2.line(frame, (border, 0), (border, args.img_height - 1), color=(169, 76, 49), thickness=10)
        video.write(frame)
        if border == args.img_width:
            continue
        else:
            border += v_pixel * direction


if __name__ == '__main__':
    main(get_args())
