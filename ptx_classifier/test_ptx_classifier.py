import os
import numpy as np
from scipy.misc import imread
import pickle
from batch_segment import batch_segment
from keraswrapper import load_model, weighted_pixelwise_crossentropy, dice_coef

np.random.seed(1)
from aid_funcs.misc import zip_save
from aid_funcs import CXRLoadNPrep as clp
from utils import *

test_path = r'C:\projects\CXR_thesis\data_repo\TEST'


def load_all_images():
    right_path = os.path.join(test_path, 'pos_cases', 'right')
    left_path = os.path.join(test_path, 'pos_cases', 'left')
    neg_path = os.path.join(test_path, 'neg_cases')
    right_files = [os.path.join(right_path, file) for file in os.listdir(right_path)]
    left_files = [os.path.join(left_path, file) for file in os.listdir(left_path)]
    neg_files = [os.path.join(neg_path, file) for file in os.listdir(neg_path)]
    nb_right = len(right_files)
    nb_left = len(left_files)
    nb_neg = len(neg_files)
    nb_test = nb_right + nb_left + nb_neg
    right_labels = np.zeros((nb_test,), dtype=np.uint8)
    left_labels = np.zeros((nb_test,), dtype=np.uint8)
    right_labels[:nb_right] = 1
    left_labels[nb_right:nb_right+nb_left] = 1
    imgs_path_lst = right_files + left_files + neg_files
    img_names = []
    images = []
    for im_count, curr_img_path in enumerate(imgs_path_lst):
        img_name = os.path.split(curr_img_path)[1][:-4]
        img = clp.load_dicom(curr_img_path)
        img = image.square_image(img)
        img = image.imresize(img, (1024, 1024))
        img_names.append(img_name)
        images.append(img)
        print("Loaded image number %i" % im_count)

    return right_labels, left_labels, img_names, images


def lung_seg_all():
    lung_seg_path = os.path.join(test_path, 'lung_seg')
    right_path = os.path.join(test_path, 'pos_cases', 'right')
    left_path = os.path.join(test_path, 'pos_cases', 'left')
    neg_path = os.path.join(test_path, 'neg_cases')
    batch_segment(right_path, lung_seg_path)
    batch_segment(left_path, lung_seg_path)
    batch_segment(neg_path, lung_seg_path)


def load_lung_masks(img_names):
    nb_images = len(img_names)
    lung_masks = np.zeros((nb_images, im_size, im_size), dtype=np.uint8)
    lung_seg_path = os.path.join(test_path, 'lung_seg')
    for i, name in enumerate(img_names):
        lung_path = os.path.join(lung_seg_path, name + '.png')
        if os.path.isfile(lung_path):
            # Loading lung mask
            curr_mask = imread(lung_path, mode='L')
            curr_mask = image.imresize(curr_mask, (im_size, im_size))
            lung_masks[i] = curr_mask
    lung_masks[lung_masks > 1] = 1
    return lung_masks


def preprocess_images(images, lung_masks):
    nb_images = len(images)
    images = np.zeros((nb_images, im_size, im_size))
    lung_masks_out =np.zeros((nb_images, im_size, im_size), dtype=np.uint8)
    for i in range(nb_images):
        bb = get_lung_bb(lung_masks[i])
        bb = np.clip(bb, 0, images[i].shape[0])
        img = images[i]
        img = img[bb[0]:bb[2], bb[1]:bb[3]]
        img = image.imresize(img.astype(np.float32), (im_size, im_size))
        lung_mask = lung_masks[i][bb[0]:bb[2], bb[1]:bb[3]]
        lung_mask = image.imresize(lung_mask, (im_size, im_size))

        img = normalize_img(img)
        images[i] = img
        lung_masks_out[i] = lung_mask


def get_ptx_scores(imgs_arr):
    '''
    '''
    from predict_ptx import fcn_model_path
    with open('class_weights_fcn_classifier.pkl', 'rb') as f:
        class_weights = pickle.load(f)
    custom_objects = {'loss': weighted_pixelwise_crossentropy(class_weights), 'dice_coef': dice_coef}
    fcn_model = load_model(fcn_model_path, custom_objects=custom_objects)

    scores = fcn_model.predict(imgs_arr, batch_size=5, verbose=1)
    scores = scores[:, :, :, 1]  # Taking only scores for ptx
    return scores

def predict_all(score_maps):
    pass


def compute_auc(predictions, right_labels, left_labels):
    '''
    todo:
    plot, show and save roc and results for right, left and total
    '''
    pass


def main():
    right_labels, left_labels, img_names, images = load_all_images()
    lung_masks = load_lung_masks(img_names)
    images = preprocess_images(images, lung_masks)
    score_maps = get_ptx_scores(images)

    predictions = predict_all(score_maps)
    compute_auc(predictions, right_labels, left_labels)

if __name__ == '__main__':
    # lung_seg_all()
    main()