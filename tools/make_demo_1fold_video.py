import argparse, os, cv2
import numpy as np
from tqdm import tqdm


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--lr_dir', type=str, default=r"D:\videoDenoise\noised000var2500")
    args.add_argument('--rec_dir', type=str, default=r"D:\videoDenoise\000")
    args.add_argument('--save_path', type=str, default=r"D:\videoDenoise")
    args.add_argument('--fps', type=int, default=30)
    args.add_argument('--img_height', type=int, default=720)
    args.add_argument('--img_width', type=int, default=1280)

    return args.parse_args()


def main(args):
    # for folder in os.listdir(args.lr_dir):

    lr_path = args.lr_dir
    rec_path = args.rec_dir
    lr_list = sorted(os.listdir(lr_path))
    rec_list = sorted(os.listdir(rec_path))
    print(lr_list)

    num_frames = len(lr_list)
    v_pixel = args.img_width // num_frames
    direction, border = 1, 0
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(os.path.join(args.save_path, 'result.mp4'), fourcc, args.fps,
                            (args.img_width, args.img_height))
    for lr_file, rec_file in tqdm(zip(lr_list, rec_list)):
        lr = cv2.imread(os.path.join(lr_path, lr_file))
        rec = cv2.imread(os.path.join(rec_path, rec_file))
        print(f'os.path.join(lr_path, lr_file):{os.path.join(lr_path, lr_file)}')
        frame = np.concatenate([rec[:, 0: border, :], lr[:, border: args.img_width, :]], axis=1)
        cv2.line(frame, (border, 0), (border, args.img_height - 1), color=(169, 76, 49), thickness=10)
        video.write(frame)
        if border == args.img_width:
            continue
        else:
            border += v_pixel * direction

    # lq = cv2.imread(r"C:\Users\D\Desktop\3354efd6751f0a0e14bb32cbe42e54f.png")
    # hq = cv2.imread(r"C:\Users\D\Desktop\c68cdb2990ee9a16b00735b5c5ba13b.png")
    # duration_time = 10
    # height, width = lq.shape[:2]

    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # video = cv2.VideoWriter(args.save_path, fourcc, args.fps, (width, height))

    # pos, num_frames = 0, args.fps * duration_time
    # v = width // num_frames
    # for step in range(num_frames):
    #     left_part = hq[:, :pos, :]
    #     right_part = lq[:, pos:, :]
    #     fusion_img = np.concatenate([left_part, right_part], axis=1)
    #     cv2.line(fusion_img, (pos, 0), (pos, height-1), color=(169, 76, 49), thickness=10)
    #     video.write(fusion_img)
    #     pos = pos + v
    #     if pos < width:
    #         pos += v

    # video.release()


if __name__ == '__main__':
    main(get_args())
