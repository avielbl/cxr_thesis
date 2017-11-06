import numpy as np
import os
from aid_funcs import image, misc

from utils import process_and_augment_data, training_path, im_size


def get_lung_masks(data_lst):
    nb_images = len(data_lst)
    lung_masks_arr = np.zeros((nb_images, 1, im_size, im_size), dtype=np.uint8)
    for i, case in enumerate(data_lst):
        lung_masks_arr[i] = image.imresize(case.lung_mask, (im_size, im_size))
    lung_masks_arr[lung_masks_arr > 0] = 1
    return lung_masks_arr


def prep_set(set):
    n = len(set)
    imgs_arr = np.zeros((n, 1, im_size, im_size))
    masks_arr = np.zeros((n, 1, im_size, im_size))
    for i, case in enumerate(set):
        imgs_arr[i] = image.imresize(case.img, (im_size, im_size))
        if case.ptx_mask is None:
            ptx_mask = np.zeros((im_size, im_size), dtype=np.uint8)
        else:
            ptx_mask = image.imresize(case.ptx_mask, (im_size, im_size))
        ptx_mask[ptx_mask > 0] = 1
        masks_arr[i] = ptx_mask
    return imgs_arr, masks_arr


if __name__ == '__main__':
    train_data_lst, val_data_lst = process_and_augment_data()
    lung_masks_arr = get_lung_masks(val_data_lst)
    train_imgs_arr, train_masks_arr = prep_set(train_data_lst)
    val_imgs_arr, val_masks_arr = prep_set(val_data_lst)
    misc.save_to_h5(lung_masks_arr, os.path.join(training_path, 'db_lung_masks_arr.h5'))
    misc.save_to_h5(train_imgs_arr, os.path.join(training_path, 'db_train_imgs_arr.h5'))
    misc.save_to_h5(train_masks_arr, os.path.join(training_path, 'db_train_masks_arr.h5'))
    misc.save_to_h5(val_imgs_arr, os.path.join(training_path, 'db_val_imgs_arr.h5'))
    misc.save_to_h5(val_masks_arr, os.path.join(training_path, 'db_val_masks_arr.h5'))
