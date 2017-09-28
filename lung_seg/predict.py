from params import seg_model_path, im_size
from utilfuncs import load_image, post_process_seg_result, pre_process_images
import numpy as np


def predict(img, model=None, verbose=0):
    """
    Function for predicting the lung segmentation of a given image.
    Usage in production:
    lung_seg_dict = predict(img_path)
    Usage in testing/evaluation:
    lung_seg_dict = predict(img_arr, model)

    :param img: either a path to the image or the image after loading and pre-processing
    :param model: (optinal) loaded model for quicker prediction
    :param verbose: (default=0) verbosity of keras predict method
    :return: dict of 2 binary masks with fields: 'r_lung_mask', 'l_lung_mask'
    """
    if model is None:
        from radfuncs.keraswrapper import load_model
        model = load_model(seg_model_path, custom_objects='dice_coef_loss')
    if isinstance(img, str):
        img = load_image(img)
        if not isinstance(img, (np.ndarray, np.generic) ):
            return -1
        img = pre_process_images(img)
    else:
        img = np.reshape(img, (1, 1, im_size, im_size))
    scores = model.predict(img, verbose=verbose)
    return post_process_seg_result(scores)