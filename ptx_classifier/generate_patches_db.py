import os
import gc

import h5py
from skimage import measure

import numpy as np

from aid_funcs.misc import zip_load, save_to_h5
from utils import *
from aid_funcs import image



def balance_classes(patches, labels):
    pos_idx, = np.where(labels == 1)
    neg_idx, = np.where(labels == 0)
    nb_pos = pos_idx.shape[0]
    nb_neg = neg_idx.shape[0]
    print('Total of {} pos patches and {} neg patches'.format(nb_pos, nb_neg))
    if nb_neg > nb_pos:
        sampled_idx = np.random.choice(range(nb_neg), nb_pos, False)
        neg_idx = [neg_idx[ind] for ind in sampled_idx]
    else:
        sampled_idx = np.random.choice(range(nb_pos), nb_neg, False)
        pos_idx = [pos_idx[ind] for ind in sampled_idx]
    all_idx = np.concatenate((pos_idx, neg_idx))
    if all_idx.size > 0:
        patches = patches[all_idx]
        labels = labels[all_idx]
    return patches, labels


def build_patches_db(set_lst, set):
    n = len(set_lst)
    patches = np.zeros((max_num_of_patches, 1, patch_sz, patch_sz), dtype=np.float32)
    labels = np.zeros((max_num_of_patches,), dtype=np.uint8)
    patches_counter = 0
    for i in range(n):
        print('Extracting pathces from case {}/{}:'.format(i, n))
        img = set_lst[i]['img']
        lung_mask = set_lst[i]['lung_mask']
        ptx_mask = set_lst[i]['ptx_mask']
        if is_ptx_case(ptx_mask):
            # Extracting all positive patches
            pos_patches = extract_patches_from_mask(img, patch_sz, ptx_mask)
            nb_pos = pos_patches['patches_count']
            patches[patches_counter:patches_counter+nb_pos] = pos_patches['patches']
            labels[patches_counter:patches_counter+nb_pos] = 1
            patches_counter += nb_pos
            # Extracting all negative patches from the lung mask minus the dilated ptx mask
            neg_mask = lung_mask.copy()
            dilated_ptx_mask = image.safe_binary_morphology(ptx_mask,sesize=np.int(patch_sz/2), mode='dilate')
            neg_mask[dilated_ptx_mask == 255] = 0
            neg_patches = extract_patches_from_mask(img, patch_sz, neg_mask)
            nb_neg = neg_patches['patches_count']
            patches[patches_counter:patches_counter+nb_neg] = neg_patches['patches']
            labels[patches_counter:patches_counter+nb_neg] = 0
            patches_counter += nb_neg
            print('Extracted {} positive patches and {} negatives'.format(nb_pos, nb_neg))
        else:
            neg_patches = extract_patches_from_mask(img, patch_sz, lung_mask)
            nb_neg = neg_patches['patches_count']
            patches[patches_counter:patches_counter + nb_neg] = neg_patches['patches']
            labels[patches_counter:patches_counter + nb_neg] = 0
            patches_counter += nb_neg
            print('Extracted {} negative patches'.format(nb_neg))
    # Removing redundant pre-allocated elements
    patches = patches[:patches_counter]
    labels = labels[:patches_counter]
    patches, labels = balance_classes(patches, labels)
    save_to_h5(patches, os.path.join(training_path, set+'_patches.h5'))
    save_to_h5(labels, os.path.join(training_path, set+'_labels.h5'))



def extract_patches_from_mask(img, patch_size, mask=None, num_of_patches=10000, stride=1, patch_pos_flag=False,
                              min_patch_mask_cover=0, plot_flag=False):
    sz = img.shape
    if patch_size % 2 == 0:
        l_support = int(patch_size / 2 - 1)
        r_support = int(patch_size / 2) + 1
    else:
        l_support = int(patch_size / 2)
        r_support = int(patch_size / 2)
    pad_size = max((r_support, l_support))
    if mask is None:
        mask = np.zeros(sz, dtype=np.uint8)
    mask_bbox = measure.regionprops(mask)
    mask_bbox = mask_bbox[0].bbox
    min_row = max((0, mask_bbox[0] - pad_size))
    max_row = min((sz[0], mask_bbox[2] + pad_size))
    min_col = max((0, mask_bbox[1] - pad_size))
    max_col = min((sz[1], mask_bbox[3] + pad_size))
    cropped_img = img[min_row:max_row,min_col:max_col]
    cropped_mask = mask[min_row:max_row,min_col:max_col]
    cropped_sz = cropped_img.shape

    patches = []
    patches_idx = []

    for r in range(pad_size+1, cropped_sz[0]-pad_size, stride):
        for c in range(pad_size+1, cropped_sz[1]-pad_size, stride):
            if not cropped_mask[r, c]:
                continue
            patch = img[r-l_support:r+r_support, c-l_support:c+r_support]
            mask_patch = cropped_mask[r-l_support:r+r_support, c-l_support:c+r_support]
            if (np.sum(mask_patch) / patch_size ** 2) < min_patch_mask_cover:
                continue
            patches.append(patch)
            patches_idx.append([r+min_row, c+min_col])

    patches_count = len(patches)
    # Randomly sample extracted patches to not exceed num_of_patches
    if patches_count > num_of_patches:
        sampled_idx = np.random.choice(range(patches_count), num_of_patches, False)
        patches_idx = [patches_idx[ind] for ind in sampled_idx]
        patches = [patches[ind] for ind in sampled_idx]
        patches_count = len(sampled_idx)
    patches = np.asanyarray(patches)
    patches = np.expand_dims(patches, axis=1)
    return {'patches': patches, 'patches_idx': patches_idx, 'patches_count': patches_count}


train_data_lst, val_data_lst = train_val_partition()
print('Extracting patches from validation images')
build_patches_db(val_data_lst, 'val')

print('Extracting patches from training images')
build_patches_db(train_data_lst, 'train')

