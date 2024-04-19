import os
import cv2
import glob
import torch
import argparse
import numpy as np

from raft import RAFT
from tqdm import tqdm
from pathlib import Path


def read_img(filename):
    img = cv2.imread(filename)
    img = torch.from_numpy(img.copy()).float().permute(2, 0, 1).cuda()
    img = img.unsqueeze(0)
    return img


def write_flow(flow, filename):
    flow = flow.permute(0, 2, 3, 1)
    flow = flow.squeeze(0).cpu().detach().numpy()

    path = os.path.dirname(filename)
    if not os.path.exists(path):
        Path(path).mkdir(parents=True, exist_ok=True)

    np.save(filename, flow)
    return


def check_img_size(x, window_size):
    _, _, h, w = x.size()
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), "constant", 0)
    return x


def generate_flow(dir_path):
    dist_list = [1] # for FMA-Net w/ T=3

    parser = argparse.ArgumentParser()
    # --model ./pretrained/raft-sintel.pth --mixed_precision
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = RAFT(args)
    model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.cuda()
    model.eval()

    dir_path = sorted(glob.glob(os.path.join(dir_path, '*')))

    for dir_name in dir_path:
        img_list = sorted(glob.glob(os.path.join(dir_name, '*.png')))
        for dist in dist_list:
            for idx in tqdm(range(len(img_list))):
                if idx < dist or idx + dist > len(img_list) - 1:
                    continue
                img0 = read_img(img_list[idx - dist])
                img1 = read_img(img_list[idx])
                img2 = read_img(img_list[idx + dist])

                _, _, h, w = img0.shape
                img0 = check_img_size(img0, window_size=8)
                img1 = check_img_size(img1, window_size=8)
                img2 = check_img_size(img2, window_size=8)

                flow1_0 = model(img1, img0)[-1]
                flow1_2 = model(img1, img2)[-1]

                flow1_0 = flow1_0[:, :, :h, :w]
                flow1_2 = flow1_2[:, :, :h, :w]

                img0_name = os.path.basename(img_list[idx - dist]).split('.')[0]
                img1_name = os.path.basename(img_list[idx]).split('.')[0]
                img2_name = os.path.basename(img_list[idx + dist]).split('.')[0]

                flow1_0_name = os.path.join(dir_name.replace('sharp', 'flow'), f'{img1_name}_{img0_name}.npy')
                flow1_2_name = os.path.join(dir_name.replace('sharp', 'flow'), f'{img1_name}_{img2_name}.npy')

                write_flow(flow1_0, flow1_0_name)
                write_flow(flow1_2, flow1_2_name)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    generate_flow('./dataset/REDS4/train_sharp_bicubic/train/train_sharp_bicubic/X4')
    generate_flow('./dataset/REDS4/val_sharp_bicubic/val/val_sharp_bicubic/X4')