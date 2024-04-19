import os
import cv2
import glob
import torch
import random

import numpy as np


def get_dataset(config, type):
    data = REDS_Dataset(config, type=type)

    if type == 'train':
        data_loader = torch.utils.data.DataLoader(data, batch_size=config.batch_size, drop_last=True, shuffle=True, num_workers=int(config.nThreads), pin_memory=True)
    elif type == 'val':
        data_loader = torch.utils.data.DataLoader(data, batch_size=1, drop_last=False, shuffle=False, num_workers=int(config.nThreads), pin_memory=True)
    elif type == 'test':
        data_loader = torch.utils.data.DataLoader(data, batch_size=1, drop_last=False, shuffle=False, num_workers=int(config.nThreads), pin_memory=True)
    else:
        raise NotImplementedError('not implemented for this mode: {}!'.format(type))

    return data_loader


class REDS_Dataset:
    def __init__(self, config, type):
        self.config = config
        self.type = type
        self.num_seq = self.config.num_seq

        bath_path = None
        if type == 'train':
            bath_path = os.path.join(config.dataset_path, 'train_blur_bicubic')
        if type == 'val':
            bath_path = os.path.join(config.dataset_path, 'val_blur_bicubic')
        if type == 'test':
            bath_path = os.path.join(config.dataset_path, 'val_blur_bicubic')

        self.seq_path = self.get_seq_path(bath_path)
        self.num_data = len(self.seq_path)

        print(f'num {type} dataset: {self.num_data}')

    def __getitem__(self, idx):
        # input
        lr_blur_path = self.seq_path[idx]
        lr_blur_seq = [cv2.imread(path) for path in lr_blur_path]
        lr_blur_seq = np.stack(lr_blur_seq, axis=0)

        if self.type == 'train' or self.type == 'val':
            # for TA loss
            lr_sharp_path = [os.path.normpath(path.replace('blur', 'sharp')) for path in lr_blur_path]
            lr_sharp_seq = [cv2.imread(path) for path in lr_sharp_path]
            lr_sharp_seq = np.stack(lr_sharp_seq, axis=0)

            # GT
            hr_sharp_path = [os.path.normpath(path.replace('blur_bicubic', 'sharp').replace('X4', '')) for path in lr_blur_path]
            hr_sharp_seq = [cv2.imread(path) for path in hr_sharp_path]
            hr_sharp_seq = np.stack(hr_sharp_seq, axis=0)

            # RAFT pseudo-GT optical flow
            flow = []
            img_c_name = os.path.basename(lr_blur_path[self.num_seq // 2]).replace('.png', '')
            for i in range(self.num_seq):
                if i == self.num_seq // 2:
                    flow.append(np.zeros_like(flow[0]))
                    continue
                filename = os.path.normpath(lr_blur_path[i].replace("blur", "flow"))
                img_name = os.path.basename(lr_blur_path[i].replace('.png', ''))
                temp = np.load(f'{filename.replace(img_name + ".png", img_c_name + "_" + img_name)}.npy')
                flow.append(temp)
            flow = np.stack(flow, axis=0)

        if self.type == 'train':
            lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow = self.get_random_patch(lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow)
            lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow = self.augment(lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow)
            return self.np2tensor(lr_blur_seq), self.np2tensor(hr_sharp_seq), self.np2tensor(lr_sharp_seq), self.flow2tensor(flow)

        if self.type == 'val':
            return self.np2tensor(lr_blur_seq), self.np2tensor(hr_sharp_seq), self.np2tensor(lr_sharp_seq), self.flow2tensor(flow)

        if self.type == 'test':
            filename = lr_blur_path[self.num_seq // 2]
            return self.np2tensor(lr_blur_seq), filename

    def get_random_patch(self, lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow):
        ih, iw, c = lr_blur_seq[0].shape

        tp = self.config.patch_size
        ip = tp // self.config.scale
        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        (tx, ty) = (self.config.scale * ix, self.config.scale * iy)

        lr_blur_seq = lr_blur_seq[:, iy:iy + ip, ix:ix + ip, :]
        hr_sharp_seq = hr_sharp_seq[:, ty:ty + tp, tx:tx + tp, :]
        lr_sharp_seq = lr_sharp_seq[:, iy:iy + ip, ix:ix + ip, :]
        flow = flow[:, iy:iy + ip, ix:ix + ip, :]

        return lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow

    def augment(self, lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow):
        # random horizontal flip
        if random.random() < 0.5:
            lr_blur_seq = lr_blur_seq[:, :, ::-1, :]
            hr_sharp_seq = hr_sharp_seq[:, :, ::-1, :]
            lr_sharp_seq = lr_sharp_seq[:, :, ::-1, :]
            flow = flow[:, :, ::-1, :]
            flow[:, :, :, 0] *= -1

        # random vertical flip
        if random.random() < 0.5:
            lr_blur_seq = lr_blur_seq[:, ::-1, :, :]
            hr_sharp_seq = hr_sharp_seq[:, ::-1, :, :]
            lr_sharp_seq = lr_sharp_seq[:, ::-1, :, :]
            flow = flow[:, ::-1, :, :]
            flow[:, :, :, 1] *= -1

        return lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow

    def np2tensor(self, x):
        # x shape: [T, H, W, C]

        # reshape to [C, T, H, W]
        ts = (3, 0, 1, 2)
        x = torch.Tensor(x.transpose(ts).astype(float)).mul_(1.0)
        # normalization [0,1]
        x = x / 255.0

        return x

    def flow2tensor(self, flow):
        # flow shape: [T, H, W, C]

        # reshape to [C, T, H, W]
        ts = (3, 0, 1, 2)
        flow = torch.Tensor(flow.transpose(ts).astype(float)).mul_(1.0)

        return flow

    def get_seq_path(self, bath_path):
        seq_list = []
        dir_list = glob.glob(os.path.join(bath_path, '*/*/*/*'))
        for dir in dir_list:
            frame_list = sorted(glob.glob(os.path.join(dir, '*.png')))
            start = (self.num_seq - 1) // 2
            end = len(frame_list) - (self.num_seq - 1) // 2
            for i in range(start, end):
                frame_seq = []
                for seq_num in range(self.num_seq):
                    frame_seq.append(frame_list[i + seq_num - start])
                seq_list.append(frame_seq)
        return seq_list

    def __len__(self):
        return self.num_data


class Custom_Dataset:
    def __init__(self, config):
        self.config = config
        self.num_seq = self.config.num_seq

        bath_path = os.path.join(config.custom_path)

        self.seq_path = self.get_seq_path(bath_path)
        self.num_data = len(self.seq_path)

        print(f'num custom dataset: {self.num_data}')

    def __getitem__(self, idx):
        # input
        lr_blur_path = self.seq_path[idx]
        lr_blur_seq = [cv2.imread(path) for path in lr_blur_path]
        lr_blur_seq = np.stack(lr_blur_seq, axis=0)

        filename = lr_blur_path[self.num_seq // 2]
        return self.np2tensor(lr_blur_seq), filename

    def np2tensor(self, x):
        # x shape: [T, H, W, C]

        # reshape to [C, T, H, W]
        ts = (3, 0, 1, 2)
        x = torch.Tensor(x.transpose(ts).astype(float)).mul_(1.0)
        # normalization [0,1]
        x = x / 255.0

        return x

    def get_seq_path(self, bath_path):
        seq_list = []
        frame_list = sorted(glob.glob(os.path.join(bath_path, '*.png')))
        start = (self.num_seq - 1) // 2
        end = len(frame_list) - (self.num_seq - 1) // 2
        for i in range(start, end):
            frame_seq = []
            for seq_num in range(self.num_seq):
                frame_seq.append(frame_list[i + seq_num - start])
            seq_list.append(frame_seq)
        return seq_list

    def __len__(self):
        return self.num_data

