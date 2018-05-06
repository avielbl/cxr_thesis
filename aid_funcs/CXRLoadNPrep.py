import pydicom as dicom
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from aid_funcs import image


def load_dicom(path):
    try:
        ds = dicom.read_file(path)
    except:
        print('File isn\'t a valid DICOM')
        return -1
    img = ds.pixel_array
    img = np.array(img)
    if ds.PhotometricInterpretation.upper() == 'MONOCHROME1':
        max_i = np.max(img)
        img = (2 ** np.ceil(np.log2(max_i - 1))) - 1 - img
    return img


def ptx_pre_process(img, lungs_mask, im_size=None, show_plots=False):
    if show_plots:
        plt.figure(0)
        plt.subplot(121)
        plt.imshow(img, 'gray')
        plt.title('before pp')

    if im_size is not None:
        curr_shape = np.shape(img)
        row_ratio = im_size / curr_shape[0]
        column_size = int(curr_shape[1] * row_ratio)
        new_sz = (im_size, column_size)
        img = misc.imresize(img, new_sz, mode='F')
    sz = np.shape(img)
    lungs_mask = misc.imresize(lungs_mask, sz)
    img = image.im_rescale(img, 0, 2 ** 8)
    lung_pixels = img[lungs_mask > 0]
    out = img
    lung_std = np.std(lung_pixels)
    out[out > max(lung_pixels)] = max(lung_pixels)
    out[out < min(lung_pixels)] = min(lung_pixels)
    out = image.im_rescale(out, -127, 127) / lung_std
    if show_plots:
        plt.subplot(122)
        plt.imshow(out, 'gray')
        plt.title('after pp')
    return img, lungs_mask
