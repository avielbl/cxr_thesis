import numpy as np
import numpy.random

from misc import zip_save

numpy.random.seed(1)

import os
from skimage import measure
from scipy.misc import imread


from aid_funcs import CXRLoadNPrep as clp
from aid_funcs import image

from utils import *

# Creating list of images
pos_path = os.path.join(training_path, 'pos_cases')
neg_path = os.path.join(training_path, 'neg_cases')
pos_files = [os.path.join(pos_path, file) for file in os.listdir(pos_path)]
neg_files = [os.path.join(neg_path, file) for file in os.listdir(neg_path)]
imgs_path_lst = pos_files + neg_files
nb_train_imgs = len(imgs_path_lst)

lung_seg_path = os.path.join(training_path, 'lung_seg_gt')
ptx_masks_path = os.path.join(training_path, 'ptx_masks_gt')
train_set_lst = []

for im_count, curr_img_path in enumerate(imgs_path_lst):
    img_name = os.path.split(curr_img_path)[1][:-4]
    img = clp.load_dicom(curr_img_path)
    img = image.square_image(img)
    lung_path = os.path.join(lung_seg_path, img_name + '.png')
    if os.path.isfile(lung_path):
        # Loading lung mask
        lung_mask = imread(lung_path, mode='L')
        lung_mask = image.imresize(lung_mask, img.shape)

        # Loading ptx mask if exist
        ptx_path = os.path.join(ptx_masks_path, img_name + '.png')
        if os.path.isfile(ptx_path):
            ptx_mask = imread(ptx_path, mode='L')
            ptx_mask = image.imresize(ptx_mask, img.shape)
            ptx_mask[lung_mask == 0] = 0
        else:
            ptx_mask = None

        # Cropping image and mask
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
        nan_img[lung_mask == 0] = np.nan
        mean_val = np.nanmean(nan_img)
        std_val = np.nanstd(nan_img)
        out_img = img.copy().astype(np.float32)
        out_img -= mean_val
        out_img /= std_val
        out_img[np.isnan(out_img)] = np.nanmax(out_img)
        train_set_lst.append({'img': out_img, 'lung_mask': lung_mask, 'ptx_mask': ptx_mask})
    print("Loaded image number %i" % im_count)


zip_save(train_set_lst, os.path.join(training_path, 'train_set.pkl'))
