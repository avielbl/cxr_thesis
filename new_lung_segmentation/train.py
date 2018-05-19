import numpy as np
np.random.seed(42)
import os
import shutil

import cv2
from keras import Model, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv2D, Dropout, MaxPooling2D, AveragePooling2D, UpSampling2D, concatenate

from aid_funcs.keraswrapper import get_unet, PlotLearningCurves, print_model_to_file
from aid_funcs.keraswrapper import dice_coef, dice_coef_loss
from keras.optimizers import Adam
import time
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import Augmentor
from keras import backend as K

from image import imresize

# K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

train_set_path = r'C:\projects\CXR_thesis\new_lung_segmentation\train_set'
val_set_path = r'C:\projects\CXR_thesis\new_lung_segmentation\val_set'
train_images_path = os.path.join(train_set_path, 'images')
train_masks_path = os.path.join(train_set_path, 'masks')
val_images_path = os.path.join(val_set_path, 'images')
val_masks_path = os.path.join(val_set_path, 'masks')
augmented_images_path = os.path.join(train_set_path, 'augmented_images')
augmented_masks_path = os.path.join(train_set_path, 'augmented_masks')
im_size = 128


def conv_block(in_layer, conv_1_filt_num, conv_2_filt_num, dropout_val_conv):
    conv_1 = Conv2D(conv_1_filt_num, (3, 3), activation='selu', dilation_rate=(1, 1), kernel_initializer='he_normal',
                    padding='same')(in_layer)
    conv_2 = Conv2D(conv_2_filt_num, (3, 3), activation='selu', dilation_rate=(1, 1), kernel_initializer='he_normal',
                    padding='same')(conv_1)
    conv_2 = Dropout(dropout_val_conv)(conv_2)
    pool_layer = MaxPooling2D(pool_size=(2, 2))(conv_2)

    return pool_layer, conv_2


def global_pooling_block(in_layer, pool_size, dropout_val, up_sampling_rate):
    global_pool = AveragePooling2D(pool_size=(pool_size, pool_size))(in_layer)
    conv_global = Conv2D(1, (1, 1), activation='selu', dilation_rate=(1, 1), kernel_initializer='he_normal',
                           padding='same')(global_pool)
    conv_global = Dropout(dropout_val)(conv_global)
    up_sampled_global = UpSampling2D(size=(up_sampling_rate, up_sampling_rate))(conv_global)

    return up_sampled_global


def create_model_psp():
    inputs = Input((im_size, im_size, 1))

    conv_1_filt_num = 32
    conv_2_filt_num = 32
    conv_3_filt_num = 32
    conv_4_filt_num = 32

    dilated_conv_1_filt_num = 64
    dilated_conv_2_filt_num = 64
    dilated_conv_3_filt_num = 64
    dilated_conv_4_filt_num = 64

    dilated_conv_1_dilatation_rate = 2
    dilated_conv_2_dilatation_rate = 4
    dilated_conv_3_dilatation_rate = 8
    dilated_conv_4_dilatation_rate = 16

    dropout_val_conv = 0.2
    dropout_val_dilated_conv = 0.2
    dropout_val_global_pooling = 0.2

    pool_1, conv_1 = conv_block(inputs, conv_1_filt_num, conv_2_filt_num, dropout_val_conv)
    pool_2, _ = conv_block(pool_1, conv_3_filt_num, conv_4_filt_num, dropout_val_conv)

    dilated_conv_1 = Conv2D(dilated_conv_1_filt_num, (3, 3), activation='selu',
                            dilation_rate=(dilated_conv_1_dilatation_rate, dilated_conv_1_dilatation_rate),
                            kernel_initializer='he_normal', padding='same')(pool_2)
    dilated_conv_2 = Conv2D(dilated_conv_2_filt_num, (3, 3), activation='selu',
                            dilation_rate=(dilated_conv_2_dilatation_rate, dilated_conv_2_dilatation_rate),
                            kernel_initializer='he_normal', padding='same')(dilated_conv_1)
    dilated_conv_3 = Conv2D(dilated_conv_3_filt_num, (3, 3), activation='selu',
                            dilation_rate=(dilated_conv_3_dilatation_rate, dilated_conv_3_dilatation_rate),
                            kernel_initializer='he_normal', padding='same')(dilated_conv_2)
    dilated_conv_4 = Conv2D(dilated_conv_4_filt_num, (3, 3), activation='selu',
                            dilation_rate=(dilated_conv_4_dilatation_rate, dilated_conv_4_dilatation_rate),
                            kernel_initializer='he_normal', padding='same')(dilated_conv_3)
    dilated_conv_4 = Dropout(dropout_val_dilated_conv)(dilated_conv_4)
    up_dilated_conv_4 = UpSampling2D(size=(4, 4))(dilated_conv_4)

    down_im_size = im_size / 4

    up_sampled_global_1 = global_pooling_block(dilated_conv_4, down_im_size, dropout_val_global_pooling, im_size)
    up_sampled_global_2 = global_pooling_block(dilated_conv_4, down_im_size / 2, dropout_val_global_pooling, im_size / 2)
    up_sampled_global_4 = global_pooling_block(dilated_conv_4, down_im_size / 4, dropout_val_global_pooling, im_size / 4)


    concatenate_layer = concatenate([conv_1, up_dilated_conv_4, up_sampled_global_1, up_sampled_global_2,
                                     up_sampled_global_4], axis=1)
    psp_out = Conv2D(64, (1, 1), activation='selu', kernel_initializer='he_normal')(concatenate_layer)
    outputs = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal')(psp_out)

    # Create model
    solver = Adam(lr=0.0001)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=solver, loss=dice_coef_loss, metrics=[dice_coef])

    return model


def augment_data():
    p = Augmentor.Pipeline(train_images_path, output_directory=augmented_images_path)
    nb = len(os.listdir(train_images_path))
    p.ground_truth(train_masks_path)
    p.rotate(probability=0.3, max_left_rotation=5, max_right_rotation=5)
    p.zoom(probability=0.3, min_factor=0.8, max_factor=1.2)
    p.random_distortion(probability=0.9, grid_width=4, grid_height=4, magnitude=4)
    p.sample(nb * 10)
    images_list = os.listdir(augmented_images_path)
    masks_list = [file for file in images_list if 'groundtruth' in file]
    for mask in masks_list:
        shutil.move(os.path.join(augmented_images_path, mask), os.path.join(augmented_masks_path, mask))


def pre_process_data(images_path, masks_path):
    images_list = os.listdir(images_path)
    masks_list = os.listdir(masks_path)
    nb_images = len(images_list)
    images_arr = np.zeros((nb_images, im_size, im_size, 1), np.float32)
    masks_arr = np.zeros((nb_images, im_size, im_size, 1), np.uint8)
    for i in range(nb_images):
        image = cv2.imread(os.path.join(images_path, images_list[i]), cv2.IMREAD_GRAYSCALE)
        image = imresize(image, (im_size, im_size))
        image_mean = np.mean(image)
        image_std = np.std(image)
        image = (image - image_mean) / image_std
        images_arr[i, :, :, 0] = image
        mask = cv2.imread(os.path.join(masks_path, masks_list[i]), cv2.IMREAD_GRAYSCALE)
        mask = imresize(mask, (im_size, im_size))
        mask[mask > 0] = 1
        masks_arr[i, :, :, 0] = mask
    return images_arr, masks_arr


def main():
    # model = create_model_psp()
    print('Creating and compiling model...')
    nb_epochs = 100
    batch_size = 20
    lr = 0.0001
    optim_fun = Adam(lr=lr)
    model = get_unet(im_size, lrelu_alpha=0.1,
            filters=32, dropout_val=0.2,
            loss_fun=dice_coef_loss, metrics=dice_coef, optim_fun=optim_fun)
    # print_model_to_file(model)

    augment_data()
    train_data, train_masks = pre_process_data(augmented_images_path, augmented_masks_path)
    val_data, val_masks = pre_process_data(val_images_path, val_masks_path)
    model_file_name = 'lung_seg_model_' + time.strftime("%H_%M_%d_%m_%Y") + '.hdf5'
    model_checkpoint = ModelCheckpoint(model_file_name, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
    reduce_lr_on_plateu = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.3, verbose=1)
    plot_curves_callback = PlotLearningCurves(metric_name='dice_coef')

    model.fit(train_data, np.uint8(train_masks), batch_size=batch_size, epochs=nb_epochs,
              verbose=1, shuffle=True, validation_data=(val_data, val_masks),
              callbacks=[model_checkpoint, early_stopping, reduce_lr_on_plateu, plot_curves_callback])


if __name__ == '__main__':
    main()
