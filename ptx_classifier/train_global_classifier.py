"""
In this script, we will do:
1. load all training set
2. load global labels per lung
3. pre-process each image and predict score map for ptx
4. train a simple cnn for predicting ptx for each lung based on score map
"""
import os
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv2D, LeakyReLU, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.applications.vgg16 import VGG16
from keras.utils import to_categorical

from aid_funcs.image import imresize, im_rescale
from aid_funcs.misc import load_from_h5, save_to_h5
from utilfuncs import seperate_lungs
from utils import *
from aid_funcs.keraswrapper import load_model, get_class_weights, weighted_pixelwise_crossentropy, dice_coef, \
    PlotLearningCurves
from prep_data_for_unet import get_lung_masks, prep_set
im_sz = 32


def prep_set_for_global_classifier():
    print('Loading data...')
    train_data_lst, val_data_lst = process_and_augment_data()
    train_imgs_arr, train_masks_arr = prep_set(train_data_lst)
    val_imgs_arr, val_masks_arr = prep_set(val_data_lst)

    db = [
        train_imgs_arr,
        train_masks_arr,
        val_imgs_arr,
        val_masks_arr
    ]

    db[0] = np.rollaxis(db[0], 1, 4)
    db[1] = np.rollaxis(db[1], 1, 4)
    db[2] = np.rollaxis(db[2], 1, 4)
    db[3] = np.rollaxis(db[3], 1, 4)

    model_name = 'U-Net_WCE'
    class_weights = get_class_weights(db[1])
    custom_objects = {'loss': weighted_pixelwise_crossentropy(class_weights), 'dice_coef': dice_coef}
    model = load_model('ptx_model_' + model_name + '.hdf5', custom_objects=custom_objects)

    train_scores_maps = model.predict(db[0], batch_size=5, verbose=1)
    train_scores_maps = train_scores_maps[:, :, :, 1]  # Taking only scores for ptx
    val_scores_maps = model.predict(db[2], batch_size=5, verbose=1)
    val_scores_maps = val_scores_maps[:, :, :, 1]  # Taking only scores for ptx

    train_l_scores, train_r_scores, train_l_labels, train_r_labels = \
        separate_maps_to_lungs(train_data_lst, train_scores_maps, db[1])
    val_l_scores, val_r_scores, val_l_labels, val_r_labels = \
        separate_maps_to_lungs(val_data_lst, val_scores_maps, db[3])

    save_to_h5((train_l_scores, train_r_scores), os.path.join(training_path, 'train_scores_maps_arr.h5'))
    save_to_h5((val_l_scores, val_r_scores), os.path.join(training_path, 'val_scores_maps_arr.h5'))
    save_to_h5((train_l_labels, train_r_labels), os.path.join(training_path, 'train_global_label_arr.h5'))
    save_to_h5((val_l_labels, val_r_labels), os.path.join(training_path, 'val_global_label_arr.h5'))


def separate_maps_to_lungs(data_lst, scores_map, labels_maps):
    lung_masks_arr = get_lung_masks(data_lst).squeeze()
    nb_items = len(data_lst)
    l_scores = np.zeros((nb_items, im_sz, im_sz, 1))
    r_scores = np.zeros((nb_items, im_sz, im_sz, 1))
    l_labels = np.zeros((nb_items,), dtype=np.uint8)
    r_labels = np.zeros((nb_items,), dtype=np.uint8)
    for i in range(nb_items):
        lung_mask = seperate_lungs(lung_masks_arr[i])
        l_scores[i], r_scores[i] = seperate_and_process_case_to_lungs(scores_map[i], lung_mask)
        l_labels[i] = np.sum(labels_maps[i].squeeze() * lung_mask.l_lung_mask)
        r_labels[i] = np.sum(labels_maps[i].squeeze() * lung_mask.r_lung_mask)

    l_labels[l_labels > 0] = 1
    r_labels[r_labels > 0] = 1

    return l_scores, r_scores, l_labels, r_labels


def seperate_and_process_case_to_lungs(score_map, lung_mask):
    left_bb = get_lung_bb(lung_mask.l_lung_mask)
    right_bb = get_lung_bb(lung_mask.r_lung_mask)
    left_map = score_map[left_bb[0]:left_bb[2], left_bb[1]:left_bb[3]]
    right_map = score_map[right_bb[0]:right_bb[2], right_bb[1]:right_bb[3]]
    left_map = imresize(left_map, (im_sz, im_sz))
    right_map = imresize(right_map, (im_sz, im_sz))
    left_map = np.expand_dims(left_map, 3)
    right_map = np.expand_dims(right_map, 3)
    return left_map, right_map


def build_model_vgg16_based(nb_epochs):
    base_model = VGG16(weights='imagenet', include_top=False,
                       input_shape=(im_sz, im_sz, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.get_layer('block4_pool').output
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    prediction = Dense(1, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=prediction)
    lr = 0.00001
    decay_fac = 1
    optim_fun = Adam(lr=lr, decay=decay_fac * lr / nb_epochs)
    from keras.optimizers import SGD
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # model.compile(loss='binary_crossentropy',
    #               optimizer=optim_fun,
    #               metrics=['accuracy'])
    return model


def build_model(nb_epochs):
    model = Sequential()

    model.add(Conv2D(32, (5, 5), padding='same', input_shape=(im_sz, im_sz, 1), kernel_initializer='he_normal'))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5), padding='same', kernel_initializer='he_normal'))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))
    lr = 0.00001
    decay_fac = 1
    optim_fun = Adam(lr=lr)
    # optim_fun = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='binary_crossentropy',
                  optimizer=optim_fun,
                  metrics=['accuracy'])
    return model


def train_model(side):

    if side == 'left':
        side_ind = 0
    else:
        side_ind = 1
    print('Loading data...')
    # for i in range(db[0].shape[0]):
    #     label = str(db[1][i])
    #     img = db[0][i].squeeze()
    #     plt.imshow(img)
    #     plt.title(label)
    #     plt.show()
    #     plt.pause(1e-3)

    # for i in range(db[0].shape[0]):
    #     db[0][i,:,:,0] = im_rescale(db[0][i].squeeze(), -127, 127)
    # for i in range(db[2].shape[0]):
    #     db[2][i,:,:,0] = im_rescale(db[2][i].squeeze(), -127, 127)

    # db[0] = np.repeat(db[0], 3, 3)
    # db[2] = np.repeat(db[2], 3, 3)
    nb_train = db[0][side_ind].shape[0]
    nb_val = db[2][side_ind].shape[0]
    train_imgs = np.zeros((nb_train, im_sz, im_sz, 1))
    val_imgs = np.zeros((nb_val, im_sz, im_sz, 1))
    for i in range(db[0][side_ind].shape[0]):
        train_imgs[i,:,:,0] = imresize(db[0][side_ind][i].squeeze(), (im_sz, im_sz))
    for i in range(db[2][side_ind].shape[0]):
        val_imgs[i,:,:,0] = imresize(db[2][side_ind][i].squeeze(), (im_sz, im_sz))
    mean_val = 0.5 #np.mean(db[0][side_ind])
    std_val = 1 #np.std(db[0][side_ind])
    db[0][side_ind] -= mean_val
    db[2][side_ind] -= mean_val
    db[0][side_ind] /= std_val
    db[2][side_ind] /= std_val

    nb_epochs = 100
    batch_size = 200
    model = build_model(nb_epochs)
    # model = build_model_vgg16_based(nb_epochs)
    model_name = 'global_scratch' + '_' + side
    # model_name = 'global_vgg16'
    model.summary()
    model_file_name = 'ptx_model_' + model_name + '.hdf5'

    model_checkpoint = ModelCheckpoint(model_file_name, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
    reduce_lr_on_plateu = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.3, verbose=1)
    plot_curves_callback = PlotLearningCurves()
    callbacks = [model_checkpoint, early_stopping, reduce_lr_on_plateu, plot_curves_callback]
    print('Start fitting...')
    model.fit(train_imgs, to_categorical(db[1][side_ind], 2), batch_size=batch_size, epochs=nb_epochs,
              validation_data=(val_imgs, to_categorical(db[3][side_ind], 2)),
              verbose=1, shuffle=True,
              callbacks=callbacks)


    print("Done!")


if __name__ == '__main__':
    # prep_set_for_global_classifier()

    db = [
        load_from_h5(os.path.join(training_path, 'train_scores_maps_arr.h5')),
        load_from_h5(os.path.join(training_path, 'train_global_label_arr.h5')).astype(np.uint8),
        load_from_h5(os.path.join(training_path, 'val_scores_maps_arr.h5')),
        load_from_h5(os.path.join(training_path, 'val_global_label_arr.h5')).astype(np.uint8)
    ]

    train_model('left')
    train_model('right')
