import numpy as np
from skimage import measure
from collections import namedtuple
np.random.seed(1)
import os
import matplotlib.pyplot as plt
from aid_funcs.plot import show_image_with_overlay
from aid_funcs.misc import zip_load
from aid_funcs import image
from training_path import training_path
model_path = r'ptx_model_13_38_30_09_2017.hdf5'

im_size = 512
patch_sz = 32
smooth = 1.
max_num_of_patches = 4000000

Case = namedtuple('Case', ['name', 'img', 'lung_mask', 'ptx_mask'])

def display_train_set():
    pos_path = os.path.join(training_path, 'pos_cases')
    neg_path = os.path.join(training_path, 'neg_cases')
    imgs_names_lst = os.listdir(pos_path) + os.listdir(neg_path)
    train_set = zip_load(os.path.join(training_path, 'train_set.pkl'))
    for i, case in enumerate(train_set):
        show_image_with_overlay(case.img, case.lung_mask, case.ptx_mask, case.name + ' ' + str(i))
        plt.pause(1e-2)
        plt.waitforbuttonpress()


def is_ptx_case(ptx_mask):
    if ptx_mask is None or np.sum(ptx_mask) == 0:
        return False
    else:
        return True


def load_data_lst():
    data_lst = zip_load(os.path.join(training_path, 'train_set.pkl'))
    return data_lst

def pre_process_case(case, nb_augmentation=0):
    # Cropping image and mask
    img = case.img
    lung_mask = case.lung_mask
    ptx_mask = case.ptx_mask

    lung_map_dilated = image.safe_binary_morphology(lung_mask, sesize=15, mode='dilate')
    lung_bbox = measure.regionprops(lung_map_dilated.astype(np.uint8))
    lung_bbox = lung_bbox[0].bbox

    img = img[lung_bbox[0]:lung_bbox[2], lung_bbox[1]:lung_bbox[3]]
    lung_mask = lung_mask[lung_bbox[0]:lung_bbox[2], lung_bbox[1]:lung_bbox[3]]
    if ptx_mask is not None:
        ptx_mask = ptx_mask[lung_bbox[0]:lung_bbox[2], lung_bbox[1]:lung_bbox[3]]

    # Resizing cropped image for getting same scale for all images
    img = image.resize_w_aspect(img.astype(np.float32), im_size, padvalue=np.nan)
    lung_mask = image.resize_w_aspect(lung_mask, im_size)
    if ptx_mask is not None:
        ptx_mask = image.resize_w_aspect(ptx_mask, im_size)

    # Normalizing each image based on mean and std of lung pixels
    nan_img = img.copy()
    nan_img = nan_img.astype(np.float32)
    # nan_img[lung_mask == 0] = np.nan
    mean_val = np.nanmean(nan_img)
    std_val = np.nanstd(nan_img)
    out_img = img.copy().astype(np.float32)
    out_img -= mean_val
    out_img /= std_val
    out_img[np.isnan(out_img)] = np.nanmax(out_img)
    return {'name': case['name'], 'img': img, 'lung_mask': lung_mask, 'ptx_mask': ptx_mask}

def crop_n_resize(img, lung_mask, bb, ptx_mask=None):
    img = img[bb[0]:bb[2], bb[1]:bb[3]]
    lung_mask = lung_mask[bb[0]:bb[2], bb[1]:bb[3]]
    if ptx_mask is not None:
        ptx_mask = ptx_mask[bb[0]:bb[2], bb[1]:bb[3]]

    # Resizing cropped image for getting same scale for all images
    img = image.resize_w_aspect(img.astype(np.float32), im_size, padvalue=np.nan)
    lung_mask = image.resize_w_aspect(lung_mask, im_size)
    if ptx_mask is not None:
        ptx_mask = image.resize_w_aspect(ptx_mask, im_size)
    return img, lung_mask, ptx_mask

def train_val_partition(data_lst=None):
    if data_lst is None:
        data_lst = load_data_lst()
    nb_train_total = len(data_lst)
    val_idx = np.random.choice(range(nb_train_total), int(0.3 * nb_train_total))

    # Partition to train and val sets
    n_val = len(val_idx)
    n_train = nb_train_total - n_val
    print('Partition to validation (n={}) and training (n={}) sets'.format(n_val, n_train))
    val_data_lst = []
    train_data_lst = []
    for i in range(nb_train_total):
        if i in val_idx:
            val_data_lst.append(data_lst[i])
        else:
            # if i not in val_idx:
            train_data_lst.append(data_lst[i])
    return train_data_lst, val_data_lst