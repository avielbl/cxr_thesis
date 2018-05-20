from scipy import ndimage
import drawnow
from image import imresize, im_rescale
from keraswrapper import load_model
import pydicom
import numpy as np
from aid_funcs.CXRLoadNPrep import load_dicom
import cv2

model_path = 'lung_seg_model.hdf5'
im_size = 128


def get_model():
    return load_model(model_path, custom_objects='dice_coef_loss')


def predict(path: str, model=None):
    if model == None:
        model = get_model()
    if path.endswith('png'):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    elif path.endswith('dcm'):
        image = load_dicom(path)
    else:
        print('Unsupported file type (only png or dcm). Aborting...')
        return None
    image = pre_process(image)
    score_map = model.predict(image)
    image = image.squeeze()
    image = imresize(image, (1024, 1024))
    image = np.uint8(im_rescale(image, 0, 255))
    mask = post_process(score_map)
    mask = imresize(mask, (1024, 1024))
    return image, mask


def pre_process(image):
    image = imresize(image, (im_size, im_size))
    image = im_rescale(image, 0, 255)
    mean_val = np.mean(image)
    std_val = np.std(image)
    image = np.reshape(image, (1, im_size, im_size, 1))
    return (image - mean_val) / std_val


def post_process(score_map):
    segm = np.uint8(score_map > 0.5)
    segm = segm.squeeze()
    segm = get_2_largest_objs(segm)
    segm = np.uint8(ndimage.binary_fill_holes(segm))
    return segm


def get_2_largest_objs(mask_raw):
    binary_mask = np.zeros_like(mask_raw, dtype=np.uint8)
    binary_mask[mask_raw > 0] = 1
    label_im, nb_labels = ndimage.label(binary_mask)

    if nb_labels > 2:
        areas = ndimage.sum(binary_mask, label_im, range(nb_labels + 1))
        sorted_areas = np.sort(areas)
        smallest_lung_ares = sorted_areas[-2]
        mask_size = areas < smallest_lung_ares
        remove_pixel = mask_size[label_im]
        label_im[remove_pixel] = 0
    out = np.zeros_like(label_im, np.uint8)
    out[label_im > 0] = 1
    return out
