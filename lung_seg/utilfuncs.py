import numpy as np
import os
import cv2
import pickle
from collections import namedtuple
from aid_funcs import image
from scipy import ndimage
from skimage import measure, morphology

from aid_funcs import CXRLoadNPrep as clp
from . import params
im_size = params.im_size

lung_masks = namedtuple('lung_masks', ['r_lung_mask', 'l_lung_mask'])

def load_images(path):
    """
    Batch loading and initial pre-process CXR images for lungs segmentation
    This doesn't include normalizing images

    :param path: path to a folder with CXR images in dicom format
    :return: images array shaped (nb_img, 1, im_size, im_size)
    """
    images_dir = os.listdir(path)
    n = len(images_dir)
    images_arr = np.ndarray((n, 1, im_size, im_size), dtype='float32')
    im_count = 0
    print("-" * 30)
    for image_name in images_dir:
        im_path = os.path.join(path, image_name)
        images_arr[im_count] = load_image(im_path)
        im_count += 1
        print("Loaded image number %i" % im_count)
    return images_arr


def load_image(im_path, pp_params_path=None):
    """
    Loading of a single CXR image for lungs segmentation

    :param im_path: path to a single CXR image in dicom format
    :param pp_params_path: (optional) path to pickled mean and std values from training to normalize the image
    :return: image array in shape (1, 1, im_size, im_size) to be used for training/ predicting
    """

    img = clp.load_dicom(im_path)
    if isinstance(img, (np.ndarray, np.generic) ):
        img = image.im_rescale(img, 0, 2 ** 16)
        img = image.resize_w_aspect(img, im_size)
        if pp_params_path is not None:
            with open(pp_params_path, 'rb') as f:
                mean_val, std_val = pickle.load(f)
            img -= mean_val
            img /= std_val
        img = np.reshape(img, (1, 1, im_size, im_size))
    return img



def load_segmentation_maps(path):
    """
    Batch loading of segmentation maps from a given folder where each segmentation map is a png file with binaty mask

    :param path: path to the folder with the images
    :return: segmentation images array shaped (nb_img, 1, im_size, im_size)
    """
    seg_map_dir = os.listdir(path)
    n = len(seg_map_dir)
    seg_map_arr = np.ndarray((n, 1, im_size, im_size), dtype='uint8')
    im_count = 0
    print("-" * 30)
    for image_name in seg_map_dir:
        seg_map = cv2.imread(os.path.join(path, image_name), cv2.IMREAD_GRAYSCALE)
        seg_map = image.im_rescale(seg_map).astype('uint8')
        seg_map = image.resize_w_aspect(seg_map, im_size)
        seg_map_arr[im_count] = seg_map
        im_count += 1
        print("Loaded segmentation image number %i" % im_count)
    return seg_map_arr


def save_pre_process_params(images_arr):
    """Function to calculate mean and std of the training set and pickle them for later use during prediction"""
    mean_val = np.mean(images_arr)
    std_val = np.std(images_arr)
    with open(params.pre_process_params_path, 'wb') as f:
        pickle.dump([mean_val, std_val], f)


def get_optimal_thresh():
    """Function to retrieve optimal threshlod for prediction scores map as calculated on the validation set"""
    with open(params.optimal_thresh_path, 'rb') as f:
        optimal_thresh = pickle.load(f)
    return optimal_thresh


def set_optimal_thresh(thresh):
    """Function for saving calculated optimal threshold for prediction scores map"""
    with open(params.optimal_thresh_path, 'wb') as f:
        pickle.dump(thresh, f)


def pre_process_images(images_arr):
    """Function to normalize array of images according to mean and std calculated on training set"""
    with open(params.pre_process_params_path, 'rb') as f:
        mean_val, std_val = pickle.load(f)
    images_arr -= mean_val
    images_arr /= std_val
    return images_arr


def seperate_lungs(seg_map):
    sz = seg_map.shape
    label_im, nb_labels = ndimage.label(seg_map)
    assert nb_labels < 3, 'More than 2 objects detected in the segmentation map'
    r_lung_mask = np.zeros_like(label_im)
    l_lung_mask = np.zeros_like(label_im)
    labels = np.unique(label_im)
    labels = labels[1:]
    for lung in labels:
        curr_lung = np.zeros_like(label_im)
        curr_lung[label_im == lung] = 1
        m = measure.moments(np.uint8(curr_lung))
        cc = m[1, 0] / m[0, 0]
        if cc < sz[1] / 2:
            r_lung_mask = curr_lung
        else:
            l_lung_mask = curr_lung
    return lung_masks(r_lung_mask, l_lung_mask)

def post_process_seg_result(scores):
    """
    Function for performing the post-process of a raw scores map
    It returns dict of 2 binary masks with fields: 'r_lung_mask', 'l_lung_mask'
    """
    scores = np.squeeze(scores)
    predicted_mask = np.zeros_like(scores)

    predicted_mask[scores >= get_optimal_thresh()] = 1
    label_im, nb_labels = ndimage.label(predicted_mask)
    # remove small objects
    if nb_labels > 2:
        areas = ndimage.sum(predicted_mask, label_im, range(nb_labels + 1))
        sorted_areas = np.sort(areas)
        smallest_lung_ares = sorted_areas[-2]
        mask_size = areas < smallest_lung_ares
        remove_pixel = mask_size[label_im]
        label_im[remove_pixel] = 0
    # closing gaps along lungs contour and fill holes
    r_lung_mask = np.zeros_like(label_im)
    l_lung_mask = np.zeros_like(label_im)
    labels = np.unique(label_im)
    labels = labels[1:]
    for lung in labels:
        curr_lung = np.zeros_like(label_im)
        curr_lung[label_im == lung] = 1
        se = morphology.disk(params.close_size)
        pad_width = ((params.close_size, params.close_size), (params.close_size, params.close_size))
        padded_mask = np.pad(curr_lung, pad_width, mode='constant')
        curr_lung = morphology.binary_closing(padded_mask, se)
        curr_lung = curr_lung[params.close_size:-params.close_size, params.close_size:-params.close_size]
        curr_lung = morphology.remove_small_holes(curr_lung, params.im_size ** 2 / 3)
        m = measure.moments(np.uint8(curr_lung))
        cc = m[1, 0] / m[0, 0]
        if cc < params.im_size / 2:
            r_lung_mask[curr_lung] = 1
        else:
            l_lung_mask[curr_lung] = 1
    return lung_masks(r_lung_mask, l_lung_mask)
