import numpy as np
np.random.seed(1)

import pickle
import matplotlib.pyplot as plt
import os
import time

from keras.models import Sequential
from keras.layers import MaxPooling2D, Dropout, Activation, Flatten, Dense, Conv2D
from keras.optimizers import SGD, rmsprop, Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils.np_utils import to_categorical

from keraswrapper import print_model_to_file, PlotLearningCurves
from aid_funcs.misc import load_from_h5
from utils import *


def get_model_for_patches():
    model = Sequential()

    model.add(Conv2D(64, (5, 5), padding='same', input_shape=(1, patch_sz, patch_sz)))
    model.add(LeakyReLU(0.01))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(LeakyReLU(0.01))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(0.01))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(LeakyReLU(0.01))
    model.add(MaxPooling2D(pool_size=(4, 4)))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(LeakyReLU(0.01))
    model.add(Dropout(0.5))

    model.add(Dense(2))
    model.add(Activation('softmax'))
    lr = 0.0001
    optim_fun = Adam(lr=lr)

    # sgd = SGD(lr=0.05, decay=0.05 / 50, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optim_fun,
                  metrics=['accuracy'])
    return model





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


def train_ptx_classifier():
    print('-' * 30)
    print('Loading data...')
    print('-' * 30)
    val_data_labels = load_from_h5(os.path.join(training_path, 'val_labels.h5'))
    val_data_patches = load_from_h5(os.path.join(training_path, 'val_patches.h5'))
    train_data_labels = load_from_h5(os.path.join(training_path, 'train_labels.h5'))
    train_data_patches = load_from_h5(os.path.join(training_path, 'train_patches.h5'))

    train_data_labels = to_categorical(train_data_labels)
    val_data_labels = to_categorical(val_data_labels)
    print('Creating and compiling model')
    nb_epochs = 100
    batch_size = 1000
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
