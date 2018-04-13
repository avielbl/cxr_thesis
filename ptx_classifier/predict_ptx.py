import os
from enum import Enum

import time

from drawnow import drawnow
from scipy import ndimage
from scipy.misc import imread
import pickle
from skimage import morphology
import numpy as np
from skimage.measure import regionprops

import image
from lung_seg.predict import predict as lung_seg_predict
from aid_funcs.keraswrapper import load_model, weighted_pixelwise_crossentropy, dice_coef
from aid_funcs import CXRLoadNPrep as clp
from plot import show_image_with_overlay
from utilfuncs import seperate_lungs
from utils import get_lung_bb, im_size, normalize_img, crop_n_resize, Case
import matplotlib.pyplot as plt


class PtxResultStruct:
    def __init__(self, l_ptx, r_ptx, l_lung_area, r_lung_area):
        self.l_ptx = l_ptx
        self.r_ptx = r_ptx
        self.l_lung_area = l_lung_area
        self.r_lung_area = r_lung_area
        self.l_coverage = 0
        self.r_coverage = 0
        self.l_result = False
        self.r_result = False
        self.generate_results()

    def generate_results(self):
        l_coverage = 100 * np.sum(self.l_ptx) / self.l_lung_area
        r_coverage = 100 * np.sum(self.r_ptx) / self.r_lung_area

        coverage_thresh = (3, 7)
        l_mask = self.l_ptx
        r_mask = self.r_ptx
        # Keeping only largest ptx as bilateral is extremely rare
        # right is positive
        if r_coverage > coverage_thresh[0]:
            # left is positive as well
            if l_coverage > coverage_thresh[0]:
                # left is dominant- keep only it
                if l_coverage > coverage_thresh[1] > r_coverage:
                    l_results = True
                    r_results = False
                    r_mask = np.zeros_like(self.r_ptx)
                else:
                    # right is dominant- keep only it
                    if r_coverage > coverage_thresh[1] > l_coverage:
                        l_results = False
                        r_results = True
                        l_mask = np.zeros_like(self.l_ptx)
                    # both sides are similar in size- report bilateral
                    else:
                        l_results = True
                        r_results = True
            # left is negative
            else:
                l_results = False
                r_results = True
                l_mask = np.zeros_like(self.l_ptx)
        # right is negative
        else:
            # left is positive
            if l_coverage > coverage_thresh[0]:
                l_results = True
                r_results = False
                r_mask = np.zeros_like(self.l_ptx)
            # both are negative
            else:
                l_results = False
                r_results = False
                r_mask = np.zeros_like(self.l_ptx)
                l_mask = np.zeros_like(self.l_ptx)
        self.l_ptx = l_mask
        self.r_ptx = r_mask
        self.l_result = l_results
        self.r_result = r_results
        self.l_coverage = l_coverage
        self.r_coverage = r_coverage


class LocalClassifier(Enum):
    FCN = 0
    PATCH = 1


class GlobalClassifier(Enum):
    DL = 0
    RULE = 1


class InputType(Enum):
    DICOM = 0
    PNG = 1


class ModelsInstance:
    def __init__(self, fcn_model_path='ptx_model_U-Net_WCE.hdf5',
                 patch_model_path='ptx_model_13_38_30_09_2017.hdf5',
                 global_classifier_model_path='',
                 lung_seg_model_path=r'C:\projects\CXR_thesis\code\lung_seg\lung_seg_model_10_17_16_02_2017.hdf5'):
        self.fcn_model_path = fcn_model_path
        self.patch_model_path = patch_model_path
        self.global_classifier_model_path = global_classifier_model_path
        self.lung_seg_model_path = lung_seg_model_path
        print('Start loading models...')
        t = time.time()
        self.load_models()
        print('Done in {} seconds'.format(time.time() - t))

    def load_models(self):
        # todo: load all models
        import keras.backend as K
        K.set_image_data_format('channels_first')
        self.lung_seg_model = load_model(self.lung_seg_model_path, custom_objects='dice_coef_loss')
        K.set_image_data_format('channels_last')
        with open('class_weights_fcn_classifier.pkl', 'rb') as f:
            class_weights = pickle.load(f)
        custom_objects = {'loss': weighted_pixelwise_crossentropy(class_weights), 'dice_coef': dice_coef}

        self.fcn_model = load_model(self.fcn_model_path,custom_objects)
        self.patch_model = None
        self.global_model = None


def predict(img_path,
            local_classifier=LocalClassifier.FCN,
            global_classifier=GlobalClassifier.RULE,
            models_instance:ModelsInstance=None):
    if models_instance is None:
        models_instance = ModelsInstance()
    image, lung_mask = preprocess(img_path, models_instance.lung_seg_model)
    print('Done preprocessing image')
    score_map = None
    if local_classifier == LocalClassifier.FCN:
        score_map = local_fcn_classifier(image, models_instance.fcn_model)
    elif local_classifier == LocalClassifier.PATCH:
        pass
    else:
        return None

    if global_classifier == GlobalClassifier.DL:
        pass
    elif global_classifier == GlobalClassifier.RULE:
        return image, global_rule_based_classify(image, score_map, lung_mask)
    else:
        return None


def batch_predict(imgs_folder_path, labels):
    models_instance = ModelsInstance()
    out_results = []
    imgs_lst = os.listdir(imgs_folder_path)
    for i, img_path in enumerate(imgs_lst):
        res = predict(os.path.join(imgs_folder_path, img_path), models_instance=models_instance)
        if labels[i][0] == 1:
            if labels[i][1] == 1:
                # bilateral
                labels_str = 'left+right'
            else:
                # only left
                labels_str = 'left'
        elif labels[i][1] == 1:
            # only right
            labels_str = 'right'
        else:
            # none
            labels_str = 'none'

        display_ptx_results(res[0], img_path, res[1], labels_str)
        out_results.append(res)
    return out_results


def load_img(input_path):
    '''
    load a cxr image from path. can handle either dicom format or png
    :param input_path:
    :param input_format:
    :return: 2d nparray of the image unprocessed
    '''
    if not os.path.isfile(input_path):
        print('File {} doesn\'t exist'.format(input_path))
        return None
    input_format = get_image_type(input_path)
    if input_format == InputType.DICOM:
        img = clp.load_dicom(input_path)
    elif input_format == InputType.PNG:
        img = imread(input_path)
    else:
        return None
    img = image.square_image(img)
    img = image.imresize(img, (1024, 1024))
    return img


def get_image_type(img_path):
    if img_path.endswith('dcm'):
        image_type = InputType.DICOM
    elif img_path.endswith('png'):
        image_type = InputType.PNG
    else:
        image_type = None
    return image_type


def preprocess(img_path, lung_seg_model=None):
    '''Performs:
    1. lung segment
    2. crop around lungs
    3. resize
    4. normalize
    '''

    res = lung_seg_predict(img_path, lung_seg_model)
    img = load_img(img_path)
    lungs_both = res.l_lung_mask + res.r_lung_mask
    lungs_both = image.imresize(lungs_both, img.shape)
    case = Case('', img, lungs_both, None)
    bb = get_lung_bb(case.lung_mask)
    img, lungs_both, _ = crop_n_resize(case, bb)
    img = normalize_img(img)


    return img, lungs_both


def global_rule_based_classify(image_in, score_map, lung_mask):

    # masking out ptx in very bright pixels (foreign objects)
    plt.figure()
    plt.subplot(331)
    plt.imshow(image_in, cmap='gray')
    plt.title('orig image')
    plt.subplot(332)
    show_image_with_overlay(image_in, lung_mask, title_str='lungs')
    plt.subplot(333)
    plt.imshow(score_map)
    plt.title('scores')

    if score_map is None:
        return None

    # thresholding scores based on roc analysis of per-pixel clacification accuracy
    ptx_map = score_map > 0.5
    plt.subplot(334)
    show_image_with_overlay(image_in, ptx_map, title_str='ptx_map')

    lung_mask_bool = lung_mask > 0
    lung_pixels = image_in[lung_mask_bool]
    mean_lung = np.mean(lung_pixels)
    std_lung = np.std(lung_pixels)
    obj_mask = image_in - mean_lung > 2 * std_lung
    obj_mask[lung_mask_bool] = False
    ptx_map[obj_mask] = 0

    plt.subplot(335)
    show_image_with_overlay(image_in, ptx_map, title_str='obj_mask')

    # morphology cleaning
    # ptx_map_closed = image.safe_binary_morphology(ptx_map, 50, 'close')
    ptx_map_noholes = morphology.remove_small_holes(ptx_map)

    sep_lungs_obj = seperate_lungs(lung_mask)
    l_ptx = ptx_map_noholes * sep_lungs_obj.l_lung_mask
    r_ptx = ptx_map_noholes * sep_lungs_obj.r_lung_mask

    l_ptx = mask_obj_far_from_edge(l_ptx, lung_mask)
    r_ptx = mask_obj_far_from_edge(r_ptx, lung_mask)

    plt.subplot(336)
    show_image_with_overlay(image_in, l_ptx+r_ptx, title_str='mask_obj_far_from_edge')

    l_lung_area = np.sum(sep_lungs_obj.l_lung_mask)
    r_lung_area = np.sum(sep_lungs_obj.r_lung_mask)

    l_ptx = morphology.remove_small_objects(l_ptx, np.round(l_lung_area / 100))
    r_ptx = morphology.remove_small_objects(r_ptx, np.round(r_lung_area / 100))

    plt.subplot(337)
    show_image_with_overlay(image_in, l_ptx + r_ptx, title_str='remove_small_objects')

    return PtxResultStruct(l_ptx, r_ptx, l_lung_area, r_lung_area)


def mask_obj_far_from_edge(ptx_mask, lung_mask) -> np.ndarray:
    sz = ptx_mask.shape
    label_im, nb_labels = ndimage.label(ptx_mask)
    lung_center = ndimage.measurements.center_of_mass(lung_mask)

    # generating lung perimeter mask
    shift_mask = np.zeros_like(lung_mask)
    if lung_center[1] < sz[1]/2: # case of right lung
        shift_mask[:-15, :-15] = lung_mask[15:,15:]
    else:
        shift_mask[:-15, 15:] = lung_mask[15:,:-15]

    lung_perim_mask = shift_mask - lung_mask
    lung_perim_mask[lung_perim_mask < 1] = 0
    lung_perim_mask = lung_perim_mask > 0 #making boolean mask
    labels = np.unique(label_im)[1:]
    # for each object, draw equivalent circle and check intersection with lung perim mask
    for label in labels:
        obj = np.zeros_like(lung_mask)
        obj[label_im == label] = 1
        center = ndimage.measurements.center_of_mass(obj)
        props = regionprops(obj)
        minor_axis = props[0]["minor_axis_length"]
        major_axis = props[0]["major_axis_length"]
        mean_diam = 0.5 * (minor_axis + major_axis)
        circle_mask = draw_circle_mask(sz, center, mean_diam)
        if np.sum(circle_mask * lung_perim_mask) == 0:
            ptx_mask[obj == 1] = 0
    return ptx_mask


def draw_circle_mask(sz:tuple, center:tuple, diam:float) -> np.ndarray:
    from skimage.draw import circle
    img = np.zeros(sz, dtype=np.uint8)
    rr, cc = circle(*center, diam/2, sz)
    img[rr, cc] = 1
    return img


def global_dl_based_classify(score_map):
    pass


def local_fcn_classifier(img, model):
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 3)
    scores = model.predict(img, verbose=0)
    return scores[:, :, :, 1].squeeze()  # Taking only scores for ptx


def validate_results(imgs, labels):
    pass


def display_ptx_results(img:np.ndarray, img_name:str, res:PtxResultStruct, label=''):
    ptx_mask = res.r_ptx + res.l_ptx
    result_title = img_name
    if res.l_result:
        result_title += ' Left PTX, size {}% '.format(res.l_coverage)
    if res.r_result:
        result_title += 'Right PTX, size {}%, '.format(res.r_coverage)
    result_title += 'gt: ' + label
    plot_out = lambda: show_image_with_overlay(img, ptx_mask, title_str=result_title)
    drawnow(plot_out)
    print(result_title)


if __name__ == '__main__':
    img_path_dicom = r"C:\projects\CXR_thesis\data_repo\TEST\clean\@@@0534_0\@@@0534_0_PA.dcm"
    img_path_png = r"C:\projects\CXR_thesis\data_repo\NIH\images\00000001_000.png"
    img, res = predict(img_path_png, models_instance=ModelsInstance())

    img_name = os.path.split(img_path_png)[1]
    display_ptx_results(img, img_name, res)

