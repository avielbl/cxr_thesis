import gc
import numpy as np
import numpy.random

from aid_funcs.misc import zip_load, zip_save
from keraswrapper import print_model_to_file, PlotLearningCurves

numpy.random.seed(1)
import pickle
import matplotlib.pyplot as plt
import os
import cv2
import time
from skimage import measure
from collections import OrderedDict

from keras.models import Model, Sequential
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout, Activation, Masking, Flatten, \
    Dense, Conv2D
from keras.optimizers import SGD, rmsprop, Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras import backend as K
import keras.callbacks

from aid_funcs import image

from utils import *


def get_model_for_patches():
    model = Sequential()

    model.add(Conv2D(16, (5, 5), padding='same', input_shape=(1, patch_sz, patch_sz)))
    model.add(LeakyReLU(0.01))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (5, 5), padding='same'))
    model.add(LeakyReLU(0.01))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(0.01))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(0.01))
    model.add(MaxPooling2D(pool_size=(4, 4)))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(LeakyReLU(0.01))
    model.add(Dropout(0.25))

    model.add(Dense(2))
    model.add(Activation('softmax'))
    lr = 0.0001
    optim_fun = Adam(lr=lr)

    # sgd = SGD(lr=0.05, decay=0.05 / 50, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optim_fun,
                  metrics=['accuracy'])
    return model


def is_ptx_case(ptx_mask):
    if ptx_mask is None or np.sum(ptx_mask) == 0:
        return False
    else:
        return True
def balance_classes(patches, labels):
    pos_idx, = np.where(labels == 1) #[i for i in range(len(labels)) if labels[i] == 1]
    neg_idx, = np.where(labels == 0) #[i for i in range(len(labels)) if labels[i] == 0]
    nb_pos = pos_idx.shape[0]
    nb_neg = neg_idx.shape[0]
    print('Total of {} pos patches and {} neg patches'.format(nb_pos, nb_neg))
    if nb_neg > nb_pos:
        sampled_idx = numpy.random.choice(range(nb_neg), nb_pos, False)
        neg_idx = [neg_idx[ind] for ind in sampled_idx]
    else:
        sampled_idx = numpy.random.choice(range(nb_pos), nb_neg, False)
        pos_idx = [pos_idx[ind] for ind in sampled_idx]
    all_idx = np.concatenate((pos_idx, neg_idx))
    if all_idx.size > 0:
        patches = patches[all_idx]
        labels = labels[all_idx]
    return patches, labels


def build_patches_db(set_lst):
    n = len(set_lst)
    patches = np.zeros((max_num_of_patches, 1, patch_sz, patch_sz), dtype=np.float32)
    labels = np.zeros((max_num_of_patches,), dtype=np.uint8)
    patches_counter = 0
    for i in range(n):
        print('Extracting pathces from case {}/{}:'.format(i, n))
        img = set_lst[i]['img']
        lung_mask = set_lst[i]['lung_mask']
        ptx_mask = set_lst[i]['ptx_mask']
        if is_ptx_case(ptx_mask):
            # Extracting all positive patches
            pos_patches = extract_patches_from_mask(img, patch_sz, ptx_mask)
            nb_pos = pos_patches['patches_count']
            patches[patches_counter:patches_counter+nb_pos] = pos_patches['patches']
            labels[patches_counter:patches_counter+nb_pos] = 1
            patches_counter += nb_pos
            # Extracting all negative patches from the lung mask minus the dilated ptx mask
            neg_mask = lung_mask.copy()
            dilated_ptx_mask = image.safe_binary_morphology(ptx_mask,sesize=np.int(patch_sz/2), mode='dilate')
            neg_mask[dilated_ptx_mask == 255] = 0
            neg_patches = extract_patches_from_mask(img, patch_sz, neg_mask)
            nb_neg = neg_patches['patches_count']
            patches[patches_counter:patches_counter+nb_neg] = neg_patches['patches']
            labels[patches_counter:patches_counter+nb_neg] = 0
            patches_counter += nb_neg
            print('Extracted {} positive patches and {} negatives'.format(nb_pos, nb_neg))
        else:
            neg_patches = extract_patches_from_mask(img, patch_sz, lung_mask)
            nb_neg = neg_patches['patches_count']
            patches[patches_counter:patches_counter + nb_neg] = neg_patches['patches']
            labels[patches_counter:patches_counter + nb_neg] = 0
            patches_counter += nb_neg
            print('Extracted {} negative patches'.format(nb_neg))
    # Removing redundant pre-allocated elements
    return patches, labels, patches_counter


def extract_patches_from_mask(img, patch_size, mask=None, num_of_patches=10000, stride=1, patch_pos_flag=False,
                              min_patch_mask_cover=0, plot_flag=False):
    sz = img.shape
    if patch_size % 2 == 0:
        l_support = int(patch_size / 2 - 1)
        r_support = int(patch_size / 2) + 1
    else:
        l_support = int(patch_size / 2)
        r_support = int(patch_size / 2)
    pad_size = max((r_support, l_support))
    if mask is None:
        mask = np.zeros(sz, dtype=np.uint8)
    mask_bbox = measure.regionprops(mask)
    mask_bbox = mask_bbox[0].bbox
    min_row = max((0, mask_bbox[0] - pad_size))
    max_row = min((sz[0], mask_bbox[2] + pad_size))
    min_col = max((0, mask_bbox[1] - pad_size))
    max_col = min((sz[1], mask_bbox[3] + pad_size))
    cropped_img = img[min_row:max_row,min_col:max_col]
    cropped_mask = mask[min_row:max_row,min_col:max_col]
    cropped_sz = cropped_img.shape

    patches = [] #np.zeros((max_num_of_patches, patch_size, patch_size))
    patches_idx = [] #np.zeros((max_num_of_patches, 2), dtype=int)

    for r in range(pad_size+1, cropped_sz[0]-pad_size, stride):
        for c in range(pad_size+1, cropped_sz[1]-pad_size, stride):
            if not cropped_mask[r, c]:
                continue
            patch = img[r-l_support:r+r_support, c-l_support:c+r_support]
            mask_patch = cropped_mask[r-l_support:r+r_support, c-l_support:c+r_support]
            if (np.sum(mask_patch) / patch_size ** 2) < min_patch_mask_cover:
                continue
            patches.append(patch)
            patches_idx.append([r+min_row, c+min_col])

    patches_count = len(patches)
    # Randomly sample extracted patches to not exceed num_of_patches
    if patches_count > num_of_patches:
        sampled_idx = numpy.random.choice(range(patches_count), num_of_patches, False)
        patches_idx = [patches_idx[ind] for ind in sampled_idx]
        patches = [patches[ind] for ind in sampled_idx]
        patches_count = len(sampled_idx)
    patches = np.asanyarray(patches)
    patches = np.expand_dims(patches, axis=1)
    return {'patches': patches, 'patches_idx': patches_idx, 'patches_count': patches_count}



# def elastic_transform(image, alpha, sigma, mask=None, random_state=None):
#     from scipy.ndimage.interpolation import map_coordinates
#     from scipy.ndimage.filters import gaussian_filter
#
#     assert len(image.shape) == 2
#
#     if random_state is None:
#         random_state = np.random.RandomState(None)
#     else:
#         random_state = np.random.RandomState(random_state)
#
#     shape = image.shape
#
#     dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant") * alpha
#     dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant") * alpha
#
#     # dx[0:10,:] = 0
#     # dx[-1:-10,:] = 0
#     # dx[:,0:10] = 0
#     # dx[:,-1:-10] = 0
#     # dy[0:10, :] = 0
#     # dy[-1:-10, :] = 0
#     # dy[:, 0:10] = 0
#     # dy[:, -1:-10] = 0
#
#     x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
#     indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
#
#     if mask is None:
#         return map_coordinates(image, indices, order=1).reshape(shape)
#     else:
#         return (
#             map_coordinates(image, indices, order=1).reshape(shape),
#             map_coordinates(mask, indices, order=1).reshape(shape))
#
#
# def rand_gamma_adjust(img, gamma_range=(0.6, 1.4)):
#     from skimage import exposure
#     from random import uniform
#     gamma = uniform(*gamma_range)
#     curr_min, curr_max = np.nanmin(img), np.nanmax(img)
#     img = image.im_rescale(img)
#     gamma_adjusted = exposure.adjust_gamma(img, gamma)
#     gamma_adjusted = image.im_rescale(gamma_adjusted, curr_min, curr_max)
#     return gamma_adjusted



# def nestedness(struct):
#     if isinstance(struct, list):
#         return max([0] + [nestedness(i) for i in struct]) + 1
#     if isinstance(struct, dict):
#         return max([0] + [nestedness(i) for i in struct.values()]) + 1
#     return 1
#
#
# def augment_data(image_arr, masks_arr, ndeformation=5, ngamma=5):
#     shape = image_arr.shape
#     shape_multiplier = ndeformation * ngamma
#     out_shape = (shape_multiplier * shape[0], shape[1], shape[2], shape[3])
#     images_arr_out = np.zeros(out_shape)
#     masks_arr_out = np.zeros(out_shape)
#     out_counter = 0
#     for im in range(shape[0]):
#         curr_image = image_arr[im].squeeze()
#         curr_mask = masks_arr[im].squeeze()
#         for i in range(ndeformation):
#             for j in range(ngamma):
#                 curr_image, curr_mask = elastic_transform(curr_image, 1024, 40, mask=curr_mask)
#                 curr_image = rand_gamma_adjust(curr_image)
#                 images_arr_out[out_counter] = curr_image
#                 masks_arr_out[out_counter] = curr_mask
#                 out_counter += 1
#     return images_arr_out, masks_arr_out


def train_ptx_classifier(prep_data=False):
    print('-' * 30)
    print('Loading data...')
    print('-' * 30)
    if prep_data:
        data_lst = zip_load(os.path.join(training_path, 'train_set.pkl'))
        nb_train_total = len(data_lst)
        val_idx = np.random.choice(range(nb_train_total), int(0.3 * nb_train_total))

        # Partition to train and val sets
        n_val = len(val_idx)
        n_train = nb_train_total - n_val
        print('Partition to validation (n={}) and training (n={}) sets'.format(n_val, n_train))
        # val_data_lst = []
        train_data_lst = []
        for i in range(nb_train_total):
            # if i in val_idx:
                # val_data_lst.append(data_lst[i])
            # else:
            if i not in val_idx:
                train_data_lst.append(data_lst[i])
        del data_lst
        gc.collect()
        # print('Extracting patches from validation images')
        # val_data_patches, val_data_labels, patches_counter = build_patches_db(val_data_lst)
        # val_data_patches = val_data_patches[:patches_counter]
        # val_data_labels = val_data_labels[:patches_counter]
        # gc.collect()
        # val_data_patches, val_data_labels = balance_classes(val_data_patches, val_data_labels)
        # pickle.dump(val_data_patches, open(os.path.join(training_path, 'val_patches.pkl'), 'wb'), protocol=4)
        # pickle.dump(val_data_labels, open(os.path.join(training_path, 'val_labels.pkl'), 'wb'), protocol=4)
        #
        # del val_data_patches
        # del val_data_labels
        # gc.collect()

        print('Extracting patches from training images')
        train_data_patches, train_data_labels, patches_counter = build_patches_db(train_data_lst)
        train_data_patches = train_data_patches[:patches_counter]
        train_data_labels = train_data_labels[:patches_counter]
        gc.collect()
        train_data_patches, train_data_labels = balance_classes(train_data_patches, train_data_labels)
        pickle.dump(train_data_patches, open(os.path.join(training_path, 'train_patches.pkl'), 'wb'), protocol=4)
        pickle.dump(train_data_labels, open(os.path.join(training_path, 'train_labels.pkl'), 'wb'), protocol=4)
        # zip_save(train_data_patches, os.path.join(training_path, 'train_patches.pkl'))
        # zip_save(train_data_labels, os.path.join(training_path, 'train_labels.pkl'))
        del train_data_patches
        del train_data_labels
        gc.collect()
    else:
        # val_data_patches = zip_load(os.path.join(training_path, 'val_patches.pkl'))
        val_data_patches = pickle.load(open(os.path.join(training_path, 'val_patches.pkl'), 'rb'))
        val_data_labels = pickle.load(open(os.path.join(training_path, 'val_labels.pkl'), 'rb'))
        train_data_patches = pickle.load(open(os.path.join(training_path, 'train_patches.pkl'), 'rb'))
        train_data_labels = pickle.load(open(os.path.join(training_path, 'train_labels.pkl'), 'rb'))
        # val_data_labels = zip_load(os.path.join(training_path, 'val_labels.pkl'))
        # train_data_patches = zip_load(os.path.join(training_path, 'train_patches.pkl'))
        # train_data_labels = zip_load(os.path.join(training_path, 'train_labels.pkl'))
    from keras.utils.np_utils import to_categorical
    train_data_labels = to_categorical(train_data_labels)
    val_data_labels = to_categorical(val_data_labels)
    print('Creating and compiling model')
    nb_epochs = 100
    batch_size = 256
    model = get_model_for_patches()

    # print_model_to_file(model)

    # Defining callbacks
    model_file_name = 'ptx_model_' + time.strftime("%H_%M_%d_%m_%Y") + '.hdf5'
    model_checkpoint = ModelCheckpoint(model_file_name, monitor='val_loss', mode='min', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1, verbose=1, mode='min')
    plot_curves_callback = PlotLearningCurves()

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    model.fit(train_data_patches, train_data_labels, batch_size=batch_size, epochs=nb_epochs, verbose=1,
              validation_data=(val_data_patches, val_data_labels), shuffle=True,
              callbacks=[plot_curves_callback, model_checkpoint, early_stopping, reduce_lr_on_plateau])
    print("Done!")


if __name__ == '__main__':
    train_ptx_classifier()
