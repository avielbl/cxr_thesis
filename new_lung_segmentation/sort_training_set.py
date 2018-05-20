import numpy as np
np.random.seed(42)
from scipy import ndimage, misc
import cv2
import os
from shutil import copy2, move
import matplotlib.pyplot as plt


def get_binary_mask(mask_raw):
    binary_mask = np.zeros_like(mask_raw, dtype=np.uint8)
    binary_mask[mask_raw == 255] = 1
    label_im, nb_labels = ndimage.label(binary_mask)
    # remove small objects
    if nb_labels > 2:
        areas = ndimage.sum(binary_mask, label_im, range(nb_labels + 1))
        sorted_areas = np.sort(areas)
        smallest_lung_ares = sorted_areas[-2]
        mask_size = areas < smallest_lung_ares
        remove_pixel = mask_size[label_im]
        label_im[remove_pixel] = 0
    out = np.zeros_like(label_im, np.uint8)
    out[label_im > 0] = 255
    return out


def main():
    orig_images_path = r'C:\projects\CXR_thesis_backup\data_repo\NIH\images'
    orig_masks_path = r'C:\projects\CXR_thesis_backup\data_repo\NIH\lung_segmentations_gt'
    dest_train_masks_path = r'C:\projects\CXR_thesis\new_lung_segmentation\train_set\masks'
    dest_val_masks_path = r'C:\projects\CXR_thesis\new_lung_segmentation\val_set\masks'
    orig_images_list = os.listdir(orig_images_path)
    orig_images_full_path_list = [os.path.join(orig_images_path, x) for x in orig_images_list]
    dest_train_images_path = r'C:\projects\CXR_thesis\new_lung_segmentation\train_set\images'
    dest_val_images_path = r'C:\projects\CXR_thesis\new_lung_segmentation\val_set\images'
    orig_masks_list = os.listdir(orig_masks_path)
    orig_masks_full_path_list = [os.path.join(orig_masks_path, x) for x in orig_masks_list]
    nb_images = len(orig_masks_list)
    val_idx = np.random.choice(range(nb_images), int(0.3 * nb_images), replace=False)

    for i, mask_file in enumerate(orig_masks_list):
        mask_raw = cv2.imread(orig_masks_full_path_list[i], cv2.IMREAD_GRAYSCALE)
        mask = get_binary_mask(mask_raw)
        # plt.subplot(121)
        # plt.imshow(mask_raw)
        # plt.subplot(122)
        # plt.imshow(mask)
        # if i in val_idx:
        #     masks_out_path = dest_val_masks_path
        #     images_out_path = dest_val_images_path
        #     set = 'val'
        # else:
        #     masks_out_path = dest_train_masks_path
        #     images_out_path = dest_train_images_path
        #     set = 'train'
        # print('moving {} case {}/ {}'.format(set, i, nb_images))
        masks_out_path = dest_train_masks_path
        images_out_path = dest_train_images_path

        cv2.imwrite(os.path.join(masks_out_path, mask_file), mask)
        image_path = os.path.join(orig_images_path, mask_file)
        copy2(image_path, images_out_path)

def make_all_masks_max():
    train_path = r'C:\projects\CXR_thesis\new_lung_segmentation\train_set'
    train_image_list = os.listdir(os.path.join(train_path, 'masks'))
    nb_train_total = len(train_image_list)
    for i in range(nb_train_total):
        file_name = train_image_list[i]
        mask = cv2.imread(os.path.join(train_path, 'masks', file_name), cv2.IMREAD_GRAYSCALE)
        mask[mask > 0] = 255
        cv2.imwrite(os.path.join(train_path, 'masks', file_name), mask)


def separate_val_from_train():
    train_path = r'C:\projects\CXR_thesis\new_lung_segmentation\train_set'
    val_path = r'C:\projects\CXR_thesis\new_lung_segmentation\val_set'
    train_image_list = os.listdir(os.path.join(train_path, 'images'))
    nb_train_total = len(train_image_list)
    val_idx = np.random.choice(range(nb_train_total), int(0.3 * nb_train_total), replace=False)
    for i in val_idx:
        image_name = train_image_list[i]
        move(os.path.join(train_path, 'images', image_name), os.path.join(val_path, 'images'))
        move(os.path.join(train_path, 'masks', image_name), os.path.join(val_path, 'masks'))

if __name__ == '__main__':
    # make_all_masks_max()
    separate_val_from_train()
    # main()
