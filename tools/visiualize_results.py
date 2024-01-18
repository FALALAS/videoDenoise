import os, cv2, argparse
import numpy as np

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--lq_dir", type=str, default=r"D:\videoDenoise\noised000var25")
    args.add_argument("--hq_dir", type=str, default=r"D:\videoDenoise\0001noise")
    args.add_argument("--save_path", type=str, default=r"D:\videoDenoise\result.avi")
    args.add_argument('--duration_time', type=float, default="3")
    args.add_argument("--fps", type=int, default=30)
    return args.parse_args()


def main(args):
    num_frames = int(args.duration_time * args.fps)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    for step, (lq, hq) in enumerate(zip(os.listdir(args.lq_dir), os.listdir(args.hq_dir))):

        lq, hq = cv2.imread(os.path.join(args.lq_dir, lq)), cv2.imread(os.path.join(args.hq_dir, hq))
        print(lq.shape)
        if step == 0:
            height, width = lq.shape[:2]
            v = width // num_frames
            video = cv2.VideoWriter(args.save_path, fourcc, args.fps, (width, height))
        pos = 0
        lq, hq = cv2.resize(lq, (width, height)), cv2.resize(hq, (width, height))
        print(lq.shape)
        for i in range(num_frames):
            left_part = hq[:, :pos, :]
            right_part = lq[:, pos:, :]
            fusion_img = np.concatenate([left_part, right_part], axis=1)
            cv2.line(fusion_img, (pos, 0), (pos, height-1), color=(169, 76, 49), thickness=10)
            video.write(fusion_img)
            pos += v
    video.release()


if __name__ == "__main__":
    main(get_args())