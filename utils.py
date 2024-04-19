import os
import sys
import cv2
import math
import torch
import numpy as np

from pathlib import Path


def write(log, str):
    sys.stdout.flush()
    log.write(str + '\n')
    log.flush()


def denorm(x):
    x = x.cpu().detach().numpy()
    x = x.clip(0, 1) * 255.0
    x = np.round(x)

    return x


def Y_PSNR(img1, img2, border=0):
    # img1 and img2 have range [0, 255]

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    diff = (img1 - img2).data.div(255)

    shave = border
    if diff.size(1) > 1:
        convert = diff.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff.mul_(convert).div_(256)
        diff = diff.sum(dim=1, keepdim=True)

    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)


def RGB_PSNR(img1, img2, border=0):
    # img1 and img2 have range [0, 255]

    img1 = img1.squeeze()
    img2 = img2.squeeze()

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    img1 = img1.permute((1, 2, 0))
    img2 = img2.permute(1, 2, 0)
    img1 = img1.cpu().detach().numpy()
    img2 = img2.cpu().detach().numpy()

    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border, ...]
    img2 = img2[border:h-border, border:w-border, ...]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# --------------------------------------------
# SSIM
# --------------------------------------------
def SSIM(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''

    img1 = img1.squeeze()
    img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    img1 = img1.permute((1, 2, 0))
    img2 = img2.permute(1, 2, 0)
    img1 = img1.cpu().detach().numpy()
    img2 = img2.cpu().detach().numpy()

    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border, ...]
    img2 = img2[border:h-border, border:w-border, ...]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def get_tOF(pre_gt_grey, gt_grey, pre_output_grey, output_grey):
    target_OF = cv2.calcOpticalFlowFarneback(pre_gt_grey, gt_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    output_OF = cv2.calcOpticalFlowFarneback(pre_output_grey, output_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    target_OF, ofy, ofx = crop_8x8(target_OF)
    output_OF, ofy, ofx = crop_8x8(output_OF)

    OF_diff = np.absolute(target_OF - output_OF)
    OF_diff = np.sqrt(np.sum(OF_diff * OF_diff, axis=-1))  # l1 vector norm

    return OF_diff.mean()


def crop_8x8(img):
    ori_h = img.shape[0]
    ori_w = img.shape[1]

    h = (ori_h // 32) * 32
    w = (ori_w // 32) * 32

    while (h > ori_h - 16):
        h = h - 32
    while (w > ori_w - 16):
        w = w - 32

    y = (ori_h - h) // 2
    x = (ori_w - w) // 2
    crop_img = img[y:y + h, x:x + w]
    return crop_img, y, x


class Report():
    def __init__(self, save_dir, type, stage):
        filename = os.path.join(save_dir, f'stage{stage}_{type}_log.txt')

        if not os.path.exists(save_dir):
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        if os.path.exists(filename):
            self.logFile = open(filename, 'a')
        else:
            self.logFile = open(filename, 'w')

    def write(self, str):
        print(str)
        write(self.logFile, str)

    def __del__(self):
        self.logFile.close()


class Train_Report():
    def __init__(self):
        self.restoration_loss = []
        self.recon_loss = []
        self.hr_warping_loss = []
        self.lr_warping_loss = []
        self.flow_loss = []
        self.D_TA_loss = []
        self.R_TA_loss = []
        self.total_loss = []
        self.psnr = []
        self.recon_psnr = []
        self.num_examples = 0

    def update(self, batch_size, restoration_loss, recon_loss, hr_warping_loss, lr_warping_loss, flow_loss, D_TA_loss, R_TA_loss, total_loss):
        self.num_examples += batch_size
        self.restoration_loss.append(restoration_loss * batch_size)
        self.recon_loss.append(recon_loss * batch_size)
        self.hr_warping_loss.append(hr_warping_loss * batch_size)
        self.lr_warping_loss.append(lr_warping_loss * batch_size)
        self.flow_loss.append(flow_loss * batch_size)
        self.D_TA_loss.append(D_TA_loss * batch_size)
        self.R_TA_loss.append(R_TA_loss * batch_size)
        self.total_loss.append(total_loss * batch_size)

    def update_restoration_metric(self, output, y):
        output = denorm(output)
        y = denorm(y)
        self.psnr.append(RGB_PSNR(output, y))

    def update_recon_metric(self, output, y):
        output = denorm(output)
        y = denorm(y)
        self.recon_psnr.append(RGB_PSNR(output, y))

    def compute_mean(self):
        self.restoration_loss = np.sum(self.restoration_loss) / self.num_examples
        self.recon_loss = np.sum(self.recon_loss) / self.num_examples
        self.hr_warping_loss = np.sum(self.hr_warping_loss) / self.num_examples
        self.lr_warping_loss = np.sum(self.lr_warping_loss) / self.num_examples
        self.flow_loss = np.sum(self.flow_loss) / self.num_examples
        self.D_TA_loss = np.sum(self.D_TA_loss) / self.num_examples
        self.R_TA_loss = np.sum(self.R_TA_loss) / self.num_examples
        self.total_loss = np.sum(self.total_loss) / self.num_examples

    def result_str(self, lr_D, lr_R, period_time):
        self.compute_mean()
        if lr_R is None:
            str = f'Recon Loss: {self.recon_loss:.6f}\tHR Warping Loss: {self.hr_warping_loss:.6f}\tFlow Loss: {self.flow_loss:.8f}\n'
            str += f'D_TA Loss: {self.D_TA_loss:.6f}\tTotal Loss: {self.total_loss:.6f}\tlearning rate: {lr_D:.7f}\tTime: {period_time:.4f}'
        else:
            str = f'Recon Loss: {self.recon_loss:.6f}\tHR Warping Loss: {self.hr_warping_loss:.6f}\tFlow Loss: {self.flow_loss:.8f}\tD_TA Loss: {self.D_TA_loss:.6f}\n'
            str += f'Restoration Loss: {self.restoration_loss:.6f}\tLR Warping Loss: {self.lr_warping_loss:.6f}\tR_TA Loss: {self.R_TA_loss:.6f}\tTotal Loss: {self.total_loss:.6f}\n'
            str += f'learning rate (D): {lr_D:.7f}\tlearning rate (R): {lr_R:.7f}\tTime: {period_time:.4f}'
        return str

    def val_result_str(self, period_time):
        self.compute_mean()
        self.psnr = np.sum(self.psnr) / self.num_examples
        self.recon_psnr = np.sum(self.recon_psnr) / self.num_examples

        if self.psnr == 0:
            str = f'Recon Loss: {self.recon_loss:.6f}\tHR Warping Loss: {self.hr_warping_loss:.6f}\tFlow Loss: {self.flow_loss:.8f}\tD_TA Loss: {self.D_TA_loss:.6f}\n'
            str += f'Total Loss: {self.total_loss:.6f}\tTime: {period_time:.4f}\n'
            str += f'Recon PSNR: {self.recon_psnr:.5f}\n'
        else:
            str = f'Recon Loss: {self.recon_loss:.6f}\tHR Warping Loss: {self.hr_warping_loss:.6f}\tFlow Loss: {self.flow_loss:.8f}\tD_TA Loss: {self.D_TA_loss:.6f}\n'
            str += f'Restoration Loss: {self.restoration_loss:.6f}\tLR Warping Loss: {self.lr_warping_loss:.6f}\tR_TA Loss: {self.R_TA_loss:.6f}\tTotal Loss: {self.total_loss:.6f}\tTime: {period_time:.4f}\n'
            str += f'Recon PSNR: {self.recon_psnr:.3f}\tPSNR: {self.psnr:.3f}\n'

        return str


class TestReport():
    def __init__(self, base_dir):
        self.base_dir = base_dir

        self.total_rgb_psnr_logFile = open(os.path.join(base_dir, 'avg_rgb_psnr.txt'), 'w')
        self.total_y_psnr_logFile = open(os.path.join(base_dir, 'avg_y_psnr.txt'), 'w')
        self.total_ssim_logFile = open(os.path.join(base_dir, 'avg_ssim.txt'), 'w')
        self.total_tOF_logFile = open(os.path.join(base_dir, 'avg_tOF.txt'), 'w')

        self.total_rgb_psnr = []
        self.total_y_psnr = []
        self.total_ssim = []
        self.total_tOF = []

        self.scene_rgb_psnr_logFile = None
        self.scene_y_psnr_logFile = None
        self.scene_ssim_logFile = None
        self.scene_tOF_logFile = None

        self.scene_rgb_psnr = None
        self.scene_y_psnr = None
        self.scene_ssim = None
        self.scene_tOF = None

        self.pre_gt_grey = None
        self.pre_output_grey = None

    def scene_init(self, scene_name):
        self.scene_rgb_psnr_logFile = open(os.path.join(self.base_dir, scene_name, scene_name + '_rgb_psnr.txt'), 'w')
        self.scene_y_psnr_logFile = open(os.path.join(self.base_dir, scene_name, scene_name + '_y_psnr.txt'), 'w')
        self.scene_ssim_logFile = open(os.path.join(self.base_dir, scene_name, scene_name + '_ssim.txt'), 'w')
        self.scene_tOF_logFile = open(os.path.join(self.base_dir, scene_name, scene_name + '_tOF.txt'), 'w')

        self.scene_rgb_psnr = []
        self.scene_y_psnr = []
        self.scene_ssim = []
        self.scene_tOF = []

    def scene_del(self, scene_name):
        write(self.scene_rgb_psnr_logFile, f'average RGB PSNR\t{np.mean(self.scene_rgb_psnr)}')
        write(self.scene_y_psnr_logFile, f'average Y PSNR\t{np.mean(self.scene_y_psnr)}')
        write(self.scene_ssim_logFile, f'average SSIM\t{np.mean(self.scene_ssim)}')
        write(self.scene_tOF_logFile, f'average tOF\t{np.mean(self.scene_tOF)}')

        write(self.total_rgb_psnr_logFile, f'{scene_name} average RGB PSNR: {np.mean(self.scene_rgb_psnr)}')
        write(self.total_y_psnr_logFile, f'{scene_name} average Y PSNR: {np.mean(self.scene_y_psnr)}')
        write(self.total_ssim_logFile, f'{scene_name} average SSIM: {np.mean(self.scene_ssim)}')
        write(self.total_tOF_logFile, f'{scene_name} average tOF: {np.mean(self.scene_tOF)}')

        self.scene_rgb_psnr_logFile.close()
        self.scene_y_psnr_logFile.close()
        self.scene_ssim_logFile.close()
        self.scene_tOF_logFile.close()

        self.scene_rgb_psnr_logFile = None
        self.scene_y_psnr_logFile = None
        self.scene_ssim_logFile = None
        self.scene_tOF_logFile = None

        self.scene_rgb_psnr = None
        self.scene_y_psnr = None
        self.scene_ssim = None
        self.scene_tOF = None

        self.pre_gt_grey = None
        self.pre_output_grey = None

    def update_metric(self, gt, output, filename):
        gt_grey = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        output_grey = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

        ts = (2, 0, 1)
        gt = torch.Tensor(gt.transpose(ts).astype(float)).mul_(1.0)
        output = torch.Tensor(output.transpose(ts).astype(float)).mul_(1.0)

        gt = gt.unsqueeze(dim=0)
        output = output.unsqueeze(dim=0)

        rgb_psnr = RGB_PSNR(output, gt, border=4)
        y_psnr = Y_PSNR(output, gt, border=4)
        ssim = SSIM(output, gt, border=4)

        self.scene_rgb_psnr.append(rgb_psnr)
        self.scene_y_psnr.append(y_psnr)
        self.scene_ssim.append(ssim)

        self.total_rgb_psnr.append(rgb_psnr)
        self.total_y_psnr.append(y_psnr)
        self.total_ssim.append(ssim)

        write(self.scene_rgb_psnr_logFile, f'{filename}\t{rgb_psnr}')
        write(self.scene_y_psnr_logFile, f'{filename}\t{y_psnr}')
        write(self.scene_ssim_logFile, f'{filename}\t{ssim}')

        if self.pre_gt_grey is not None:
            tOF = get_tOF(self.pre_gt_grey, gt_grey, self.pre_output_grey, output_grey)
            self.scene_tOF.append(tOF)
            self.total_tOF.append(tOF)
            write(self.scene_tOF_logFile, f'{filename}\t{tOF}')

        self.pre_gt_grey = gt_grey
        self.pre_output_grey = output_grey

    def __del__(self):
        write(self.total_rgb_psnr_logFile, f'total average RGB PSNR: {np.mean(self.total_rgb_psnr)}')
        write(self.total_y_psnr_logFile, f'total average Y PSNR: {np.mean(self.total_y_psnr)}')
        write(self.total_ssim_logFile, f'total average SSIM: {np.mean(self.total_ssim)}')
        write(self.total_tOF_logFile, f'total average tOF: {np.mean(self.total_tOF)}')

        self.total_rgb_psnr_logFile.close()
        self.total_y_psnr_logFile.close()
        self.total_ssim_logFile.close()
        self.total_tOF_logFile.close()


class SaveManager():
    def __init__(self, config):
        self.config = config

    def save_batch_images(self, src, batch_size, step):
        num = 5 if batch_size > 5 else batch_size
        dir = self.config.log_dir
        filename = os.path.join(dir, f'{step:08d}.png')
        scale = self.config.scale

        c, h, w = src[-1][0].shape
        log_img = np.zeros((c, h * num, w * len(src)), dtype=np.uint8)
        for i in range(num):
            for j in range(len(src)):
                tmp = denorm(src[j][i])
                if tmp.shape[1] < h:
                    tmp = np.transpose(tmp, (1, 2, 0))
                    tmp = cv2.resize(tmp, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    tmp = np.transpose(tmp, (2, 0, 1))
                log_img[:, i * h:(i + 1) * h, j * w:(j + 1) * w] = tmp

        self.save_image(log_img, filename)

    def save_image(self, src, filename):
        path = os.path.dirname(filename)
        if not os.path.exists(path):
            Path(path).mkdir(parents=True, exist_ok=True)
        src = np.transpose(src, (1, 2, 0))
        cv2.imwrite(filename, src)
