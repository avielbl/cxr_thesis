import numpy as np
from keras.losses import binary_crossentropy, categorical_crossentropy

np.random.seed(1)
from collections import namedtuple
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from skimage import measure
from scipy.misc import imread

from aid_funcs import image
from aid_funcs.misc import zip_save, save_to_h5, load_from_h5
from aid_funcs.keraswrapper import get_unet, load_model, plot_first_layer, PlotLearningCurves, get_class_weights, \
    weighted_pixelwise_crossentropy, dice_coef_loss, dice_coef
from lung_seg.utilfuncs import seperate_lungs
from aid_funcs import CXRLoadNPrep as clp
from utils import *

Case = namedtuple('Case', ['left', 'right'])
Data = namedtuple('Case', ['img', 'lung_mask', 'ptx_mask'])

EPSILON = 1e-8

# def generate_data_lst():
#     lung_seg_path = os.path.join(training_path, 'lung_seg_gt')
#     ptx_masks_path = os.path.join(training_path, 'ptx_masks_gt')
#     imgs_path_lst = create_imgs_path_lst()
#     train_set_lst = []
#
#     for im_count, curr_img_path in enumerate(imgs_path_lst):
#         img_name = os.path.split(curr_img_path)[1][:-4]
#         img = clp.load_dicom(curr_img_path)
#         img = image.square_image(img)
#         lung_path = os.path.join(lung_seg_path, img_name + '.png')
#         if os.path.isfile(lung_path):
#             # Loading lung mask and change its size to match img size
#             lungs_mask = imread(lung_path, mode='L')
#             lungs_mask = image.imresize(lungs_mask, img.shape)
#
#             # Loading ptx mask if exist and change its size to match img size
#             ptx_path = os.path.join(ptx_masks_path, img_name + '.png')
#             if os.path.isfile(ptx_path):
#                 ptx_mask = imread(ptx_path, mode='L')
#                 ptx_mask = image.imresize(ptx_mask, img.shape)
#                 ptx_mask[lungs_mask == 0] = 0
#             else:
#                 ptx_mask = None
#
#             case = get_lungs_bb(img, lungs_mask, ptx_mask)
#             case = normalize_case(case)
#             train_set_lst.append(case)
#         print("Loaded image number %i" % im_count)
#     return train_data_lst


def create_imgs_path_lst():
    # Creating list of images' paths
    pos_path = os.path.join(training_path, 'pos_cases')
    neg_path = os.path.join(training_path, 'neg_cases')
    pos_files = [os.path.join(pos_path, file) for file in os.listdir(pos_path)]
    neg_files = [os.path.join(neg_path, file) for file in os.listdir(neg_path)]
    return pos_files + neg_files


def get_lungs_bb(img, lungs_mask, ptx_mask=None):
    def get_lung_bb(img, lung_mask, ptx_mask=None):
        lung_bbox = measure.regionprops(lung_mask.astype(np.uint8))
        lung_bbox = lung_bbox[0].bbox
        img = img[lung_bbox[0]:lung_bbox[2], lung_bbox[1]:lung_bbox[3]]
        lung_mask = lung_mask[lung_bbox[0]:lung_bbox[2], lung_bbox[1]:lung_bbox[3]]
        if ptx_mask is not None:
            ptx_mask = ptx_mask[lung_bbox[0]:lung_bbox[2], lung_bbox[1]:lung_bbox[3]]
        return Data(img, lung_mask, ptx_mask)

    lung_map_dilated = image.safe_binary_morphology(lungs_mask, sesize=15, mode='dilate')
    sep_lung_masks = seperate_lungs(lung_map_dilated)
    left = get_lung_bb(img, sep_lung_masks.l_lung_mask, ptx_mask)
    right = get_lung_bb(img, sep_lung_masks.r_lung_mask, ptx_mask)
    return Case(left, right)


def normalize_case(case):
    def resize_side(side):
        # Resizing cropped image and masks for getting same scale for all images
        img = image.imresize(side.img.astype(np.float32), (im_size, im_size))
        lung_mask = image.imresize(side.lung_mask.astype(np.float32), (im_size, im_size)).astype(np.uint8)
        if side.ptx_mask is not None:
            ptx_mask = image.imresize(side.ptx_mask.astype(np.float32), (im_size, im_size)).astype(np.uint8)
        else:
            ptx_mask = None
        return Data(img, lung_mask, ptx_mask)

    def normalize_img(img, lung_mask):
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
        return out_img

    left = resize_side(case.left)
    right = resize_side(case.right)
    l_img = normalize_img(left.img, left.lung_mask)
    r_img = normalize_img(right.img, right.lung_mask)
    return Case(Data(l_img, left.lung_mask, left.ptx_mask), Data(r_img, right.lung_mask, right.ptx_mask))


def prep_data(train_data_lst, val_data_lst, side_str):
    def prep_set(set):
        n = len(set)
        imgs_arr = np.zeros((n, 1, im_size, im_size))
        masks_arr = np.zeros((n, 1, im_size, im_size))
        for i in range(n):
            if side_str == 'left':
                case = set[i].left
            elif side_str == 'right':
                case = set[i].right
            else:
                raise ValueError()

            imgs_arr[i] = image.imresize(case.img, (im_size, im_size))
            if case.ptx_mask is None:
                ptx_mask = np.zeros((im_size, im_size), dtype=np.uint8)
            else:
                ptx_mask = image.imresize(case.ptx_mask, (im_size, im_size))
            ptx_mask[ptx_mask > 0] = 1
            masks_arr[i] = ptx_mask
        return imgs_arr, masks_arr

    train_imgs_arr, train_masks_arr = prep_set(train_data_lst)
    val_imgs_arr, val_masks_arr = prep_set(val_data_lst)
    db = (train_imgs_arr, train_masks_arr, val_imgs_arr, val_masks_arr)
    return db


def get_lung_masks(data_lst):
    nb_images = len(data_lst)
    lung_masks_arr = np.zeros((nb_images, 1, im_size, im_size), dtype=np.uint8)
    for i, case in enumerate(data_lst):
        lung_masks_arr[i] = image.imresize(case.lung_mask, (im_size, im_size))
    lung_masks_arr[lung_masks_arr > 0] = 1
    return lung_masks_arr


def train_model(db, model_name, class_weights=(1, 1)):
    print('Building model...')
    optim_fun = Adam(lr=0.00001, decay=0.00002)
    # loss_fun = weighted_pixelwise_crossentropy(class_weights)
    loss_fun = dice_coef_loss
    metrics = dice_coef
    model = get_unet(im_size, filters=16, optim_fun=optim_fun,
                     loss_fun=loss_fun,
                     metrics=dice_coef,
                     nb_classes=1)
    model.summary()
    model_file_name = 'ptx_model_' + model_name + '.hdf5'

    model_checkpoint = ModelCheckpoint(model_file_name, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
    reduce_lr_on_plateu = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=3, factor=0.3, verbose=1)
    # plot_curves_callback = PlotLearningCurves()
    plot_curves_callback = PlotLearningCurves(metric_name='dice_coef')
    callbacks = [model_checkpoint, early_stopping, reduce_lr_on_plateu, plot_curves_callback]
    # print('Transform masks to one-hot...')
    # db[1] = categorize(db[1])
    # db[3] = categorize(db[3])
    print('Start fitting...')
    model.fit(db[0], db[1], batch_size=5, epochs=100,
              validation_data=(db[2], db[3]),
              verbose=1, shuffle=True,
              callbacks=callbacks,
              class_weight=class_weights)

    return model

def categorize(arr):
    out = np.zeros((arr.shape[0], 2, arr.shape[2], arr.shape[3]), dtype=np.uint8)
    out[:, 0, :, :] = (1 - arr).squeeze()
    out[:, 1, :, :] = arr.squeeze()
    return out

##########################################################
###########            MAIN SCRIPT        ################
##########################################################
print('Loading data...')
db = [
    load_from_h5(os.path.join(training_path, 'db_train_imgs_arr.h5')),
    load_from_h5(os.path.join(training_path, 'db_train_masks_arr.h5')).astype(np.uint8),
    load_from_h5(os.path.join(training_path, 'db_val_imgs_arr.h5')),
    load_from_h5(os.path.join(training_path, 'db_val_masks_arr.h5')).astype(np.uint8)
]

# train_shape = db[1].shape
# nb_pixels = train_shape[0] * train_shape[1] * train_shape[2] * train_shape[3]
# nb_pos = np.sum(db[1])
# nb_neg = nb_pixels - nb_pos

# class_weights = get_class_weights(db[1])
class_weights = None
model_name = 'U-Net_DICE'
# model = train_model(db, model_name, class_weights)
from batch_predict_unet import analyze_performance
analyze_performance(model=None, val_data=(db[2], db[3]), model_name=model_name)