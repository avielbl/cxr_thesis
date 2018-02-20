import os
from enum import Enum

import time
from scipy.misc import imread
import pickle
import image
from lung_seg.predict import predict as lung_seg_predict
from aid_funcs.keraswrapper import load_model, weighted_pixelwise_crossentropy, dice_coef
from aid_funcs import CXRLoadNPrep as clp
from utils import get_lung_bb, im_size, normalize_img


class LocalClassifier(Enum):
    FCN = 0
    PATCH = 1


class GlobalClassifier(Enum):
    DL = 0
    RULE = 1


class InputType(Enum):
    DICOM = 0
    PNG = 1


class ModelsInstance():
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
    image = preprocess(img_path, models_instance.lung_seg_model)
    print('Done preprocessing image')
    if local_classifier == LocalClassifier.FCN:
        score_map = local_fcn_classifier(image, models_instance.fcn_model)
        plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.subplot(122)
        plt.imshow(score_map)
        plt.show()
        pass
    elif local_classifier == LocalClassifier.PATCH:
        pass
    else:
        return None

    if global_classifier == GlobalClassifier.DL:
        pass
    elif global_classifier == GlobalClassifier.RULE:
        pass
    else:
        return None



def batch_predict():
    pass


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
    from aid_funcs.plot import show_image_with_overlay
    from aid_funcs.image import imresize
    img = load_img(img_path)
    img = imresize(img, (256, 256))
    lungs_both = res.l_lung_mask + res.r_lung_mask
    show_image_with_overlay(img, lungs_both)
    bb = get_lung_bb(lungs_both)
    bb = np.clip(bb, 0, img.shape[0])
    img = img[bb[0]:bb[2], bb[1]:bb[3]]
    img = image.imresize(img.astype(np.float32), (im_size, im_size))
    img = normalize_img(img)
    return img


def global_rule_based_classify(score_map):
    pass


def global_dl_based_classify(score_map):
    pass


def local_fcn_classifier(img, model):
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 3)
    scores = model.predict(img, verbose=0)
    return scores[:, :, :, 1].squeeze()  # Taking only scores for ptx


def validate_results(imgs, labels):
    pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    img_path_dicom = r"C:\projects\CXR_thesis\data_repo\TEST\clean\@@@0534_0\@@@0534_0_PA.dcm"
    img_path_png = r"C:\projects\CXR_thesis\data_repo\NIH\images\00000001_000.png"
    models_instance = ModelsInstance()
    predict(img_path_dicom, models_instance=models_instance)
