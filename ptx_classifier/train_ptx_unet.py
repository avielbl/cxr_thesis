import numpy as np

np.random.seed(1)
from collections import namedtuple
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc
from skimage import measure
from scipy.misc import imread

from aid_funcs import image
from aid_funcs.misc import zip_save, save_to_h5, load_from_h5
from aid_funcs.keraswrapper import get_unet, load_model, plot_first_layer, PlotLearningCurves
from lung_seg.utilfuncs import seperate_lungs
from aid_funcs import CXRLoadNPrep as clp
from utils import *

Case = namedtuple('Case', ['left', 'right'])
Data = namedtuple('Case', ['img', 'lung_mask', 'ptx_mask'])


def generate_data_lst():
    lung_seg_path = os.path.join(training_path, 'lung_seg_gt')
    ptx_masks_path = os.path.join(training_path, 'ptx_masks_gt')
    imgs_path_lst = create_imgs_path_lst()
    train_set_lst = []

    for im_count, curr_img_path in enumerate(imgs_path_lst):
        img_name = os.path.split(curr_img_path)[1][:-4]
        img = clp.load_dicom(curr_img_path)
        img = image.square_image(img)
        lung_path = os.path.join(lung_seg_path, img_name + '.png')
        if os.path.isfile(lung_path):
            # Loading lung mask and change its size to match img size
            lungs_mask = imread(lung_path, mode='L')
            lungs_mask = image.imresize(lungs_mask, img.shape)

            # Loading ptx mask if exist and change its size to match img size
            ptx_path = os.path.join(ptx_masks_path, img_name + '.png')
            if os.path.isfile(ptx_path):
                ptx_mask = imread(ptx_path, mode='L')
                ptx_mask = image.imresize(ptx_mask, img.shape)
                ptx_mask[lungs_mask == 0] = 0
            else:
                ptx_mask = None

            case = get_lungs_bb(img, lungs_mask, ptx_mask)
            case = normalize_case(case)
            train_set_lst.append(case)
        print("Loaded image number %i" % im_count)
    return train_data_lst


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


# TODO: update commented 2 functions bellow to fit partition to 2 lungs
# def get_lung_masks(data_lst):
#     nb_images = len(data_lst)
#     lung_masks_arr = np.zeros((nb_images, 1, im_size, im_size), dtype=np.uint8)
#     for i, case in enumerate(data_lst):
#         lung_masks_arr[i] = image.imresize(case['lung_mask'], (im_size, im_size))
#     lung_masks_arr[lung_masks_arr > 0] = 1
#     return lung_masks_arr
#
# def augment_data(db):
#     pass
#
# def analyze_performance(db):
#     model_path = 'ptx_model_unet' + '.hdf5'
#     model = load_model(model_path, custom_objects='dice_coef_loss')
#     lung_masks_arr = get_lung_masks(val_data_lst)
#     ptx_pred = model.predict(db[2], batch_size=10, verbose=1)
#     ptx_pred *= lung_masks_arr
#     # calculating ROC per pixel
#     fpr, tpr, thresh = roc_curve(db[3].flatten(), ptx_pred.flatten())
#     roc_auc = auc(fpr, tpr)
#     dist_to_opt = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
#     opt_ind = np.argmin(dist_to_opt)
#     opt_thresh = thresh[opt_ind]
#
#     # plotting the roc
#     plt.figure(1)
#     plt.plot(fpr, tpr, label='ROC')
#     # plt.plot(fpr, thresh, label='Threshold')
#     plt.plot(fpr[opt_ind], tpr[opt_ind], 'ro', label='Optimal thresh')
#     plt.minorticks_on()
#     plt.grid(b=True, which='both')
#     plt.legend(loc='upper right')
#     plt.title('ROC curve (area = %0.2f, opt thresh = %0.2f)' % (100 * roc_auc, opt_thresh))
#     plt.savefig('roc analysis unet.png')



def train_model(db):
    lr = 0.0001
    optim_fun = Adam(lr=0.00001, decay=0.00002)

    model = get_unet(im_size, filters=16, optim_fun=optim_fun)
    model.summary()
    model_file_name = 'ptx_model_unet' + '.hdf5'

    model_checkpoint = ModelCheckpoint(model_file_name, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
    reduce_lr_on_plateu = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.3, verbose=1)
    plot_curves_callback = PlotLearningCurves(metric_name='dice_coef')
    callbacks = [model_checkpoint, early_stopping, reduce_lr_on_plateu, plot_curves_callback]
    model.fit(db[0], db[1], batch_size=5, epochs=100,
              validation_data=(db[2], db[3]),
              verbose=1, shuffle=True,
              callbacks=callbacks)


##########################################################
###########            MAIN SCRIPT        ################
##########################################################
db = [
    load_from_h5(os.path.join(training_path, 'db_train_imgs_arr.h5')),
    load_from_h5(os.path.join(training_path, 'db_train_masks_arr.h5')).astype(np.uint8),
    load_from_h5(os.path.join(training_path, 'db_val_imgs_arr.h5')),
    load_from_h5(os.path.join(training_path, 'db_val_masks_arr.h5')).astype(np.uint8)
]

model_left = train_model(db)
# analyze_performance(db)
