import os
import glob
import shutil

from pathlib import  Path


def move_to_train(reds4_dataset_path, dir_name, p):
    filelist = sorted(glob.glob(os.path.join(reds4_dataset_path, dir_name, p)))
    for idx, val_dir in enumerate(filelist):
        dirname = 240 + idx
        shutil.move(val_dir, os.path.join(os.path.dirname(val_dir.replace('val', 'train')), f'{dirname}'))


def move_to_val(reds4_dataset_path, reds4_val_list, dir_name, p):
    for val_dir in reds4_val_list:
        dirname = glob.glob(os.path.join(reds4_dataset_path, dir_name, p, val_dir))[0]
        shutil.move(dirname, dirname.replace('train', 'val'))


if __name__ == '__main__':
    reds_dataset_path = './dataset/REDS'
    reds4_dataset_path = './dataset/REDS4'
    if not os.path.exists(reds4_dataset_path):
        Path(reds4_dataset_path).mkdir(parents=True, exist_ok=True)

    reds4_val_list = ['000', '011', '015', '020']

    shutil.copytree(os.path.join(reds_dataset_path, 'train_sharp'), os.path.join(reds4_dataset_path, 'train_sharp'))
    shutil.copytree(os.path.join(reds_dataset_path, 'train_blur'), os.path.join(reds4_dataset_path, 'train_blur'))
    shutil.copytree(os.path.join(reds_dataset_path, 'train_sharp_bicubic'), os.path.join(reds4_dataset_path, 'train_sharp_bicubic'))
    shutil.copytree(os.path.join(reds_dataset_path, 'train_blur_bicubic'), os.path.join(reds4_dataset_path, 'train_blur_bicubic'))

    shutil.copytree(os.path.join(reds_dataset_path, 'val_sharp'), os.path.join(reds4_dataset_path, 'val_sharp'))
    shutil.copytree(os.path.join(reds_dataset_path, 'val_blur'), os.path.join(reds4_dataset_path, 'val_blur'))
    shutil.copytree(os.path.join(reds_dataset_path, 'val_sharp_bicubic'), os.path.join(reds4_dataset_path, 'val_sharp_bicubic'))
    shutil.copytree(os.path.join(reds_dataset_path, 'val_blur_bicubic'), os.path.join(reds4_dataset_path, 'val_blur_bicubic'))

    move_to_train(reds4_dataset_path, 'val_sharp', '*/*/*')
    move_to_train(reds4_dataset_path, 'val_blur', '*/*/*')
    move_to_train(reds4_dataset_path, 'val_sharp_bicubic', '*/*/*/*')
    move_to_train(reds4_dataset_path, 'val_blur_bicubic', '*/*/*/*')

    move_to_val(reds4_dataset_path, reds4_val_list, 'train_sharp', '*/*')
    move_to_val(reds4_dataset_path, reds4_val_list, 'train_blur', '*/*')
    move_to_val(reds4_dataset_path, reds4_val_list, 'train_sharp_bicubic', '*/*/*')
    move_to_val(reds4_dataset_path, reds4_val_list, 'train_blur_bicubic', '*/*/*')
