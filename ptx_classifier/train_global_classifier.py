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
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16

from image import imresize
from misc import load_from_h5, save_to_h5
from utils import *
from aid_funcs.keraswrapper import load_model, get_class_weights, weighted_pixelwise_crossentropy, dice_coef, \
    PlotLearningCurves

im_sz = 32
def prep_set_for_global_classifier():
    print('Loading data...')
    db = [
        load_from_h5(os.path.join(training_path, 'db_train_imgs_arr.h5')),
        load_from_h5(os.path.join(training_path, 'db_train_masks_arr.h5')).astype(np.uint8),
        load_from_h5(os.path.join(training_path, 'db_val_imgs_arr.h5')),
        load_from_h5(os.path.join(training_path, 'db_val_masks_arr.h5')).astype(np.uint8)
    ]
    db[0] = np.rollaxis(db[0], 1, 4)
    db[1] = np.rollaxis(db[1], 1, 4)
    db[2] = np.rollaxis(db[2], 1, 4)
    db[3] = np.rollaxis(db[3], 1, 4)

    nb_train = db[0].shape[0]
    nb_val = db[3].shape[0]

    train_global_label = np.sum(db[1], axis=(1, 2, 3))
    train_global_label[train_global_label > 0] = 1
    val_global_label = np.sum(db[3], axis=(1, 2, 3))
    val_global_label[val_global_label > 0] = 1

    model_name = 'U-Net_WCE'
    class_weights = get_class_weights(db[1])
    custom_objects={'loss': weighted_pixelwise_crossentropy(class_weights), 'dice_coef':dice_coef}
    model = load_model('ptx_model_' + model_name + '.hdf5', custom_objects=custom_objects)

    train_scores_maps = model.predict(db[0], batch_size=5, verbose=1)
    train_scores_maps = train_scores_maps[:, :, :, 1]  # Taking only scores for ptx
    val_scores_maps = model.predict(db[2], batch_size=5, verbose=1)
    val_scores_maps = val_scores_maps[:, :, :, 1]  # Taking only scores for ptx

    #normalizing data
    mean_val = np.mean(train_scores_maps)
    std_val = np.std(train_scores_maps)
    train_scores_maps = (train_scores_maps - mean_val) / std_val
    val_scores_maps = (val_scores_maps - mean_val) / std_val

    train_scores_maps_resized = np.zeros((nb_train, im_sz, im_sz, 1))
    for i in range(nb_train):
        train_scores_maps_resized[i] = np.expand_dims(imresize(train_scores_maps[i], (im_sz, im_sz)), 3)

    val_scores_maps_resized = np.zeros((nb_val, im_sz, im_sz, 1))
    for i in range(nb_val):
        val_scores_maps_resized[i] = np.expand_dims(imresize(val_scores_maps[i], (im_sz, im_sz)), 3)
    save_to_h5(train_scores_maps_resized, os.path.join(training_path, 'train_scores_maps_arr.h5'))
    save_to_h5(val_scores_maps_resized, os.path.join(training_path, 'val_scores_maps_arr.h5'))
    save_to_h5(train_global_label, os.path.join(training_path, 'train_global_label_arr.h5'))
    save_to_h5(val_global_label, os.path.join(training_path, 'val_global_label_arr.h5'))


def build_model_vgg16_based(nb_epochs):
    base_model = VGG16(weights='imagenet', include_top=False,
                       input_shape=(im_sz, im_sz, 1))
    x = base_model.get_layer('block4_pool').output
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    prediction = Dense(1, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=prediction)
    lr = 0.00001
    decay_fac = 1
    optim_fun = Adam(lr=lr, decay=decay_fac * lr / nb_epochs)

    model.compile(loss='binary_crossentropy',
                  optimizer=optim_fun,
                  metrics=['accuracy'])
    return model


def build_model(nb_epochs):
    model = Sequential()

    model.add(Conv2D(16, (3, 3), padding='same', input_shape=(im_sz, im_sz, 1), kernel_initializer='he_normal'))
    model.add(LeakyReLU(0.0))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(LeakyReLU(0.01))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(LeakyReLU(0.0))
    # model.add(Dropout(0.5))

    model.add(Dense(1, activation='softmax'))
    lr = 0.00001
    decay_fac = 1
    optim_fun = Adam(lr=lr, decay=decay_fac * lr / nb_epochs)

    model.compile(loss='binary_crossentropy',
                  optimizer=optim_fun,
                  metrics=['accuracy'])
    return model


def train_model():
    # prep_set_for_global_classifier()
    print('Loading data...')
    db = [
        load_from_h5(os.path.join(training_path, 'train_scores_maps_arr.h5')),
        load_from_h5(os.path.join(training_path, 'train_global_label_arr.h5')),
        load_from_h5(os.path.join(training_path, 'val_scores_maps_arr.h5')).astype(np.uint8),
        load_from_h5(os.path.join(training_path, 'val_global_label_arr.h5')).astype(np.uint8)
    ]
    db[0] = np.repeat(db[0], 3, 3)
    db[2] = np.repeat(db[0], 3, 3)
    # for i in range(db[0].shape[0]):
    #     label = str(db[1][i])
    #     img = db[0][i].squeeze()
    #     plt.imshow(img)
    #     plt.title(label)
    #     plt.show()
    #     plt.pause(1e-3)
    nb_epochs = 100
    batch_size = 100
    # model = build_model(nb_epochs)
    model = build_model_vgg16_based(nb_epochs)
    model_name = 'global_vgg16'
    model.summary()
    model_file_name = 'ptx_model_' + model_name + '.hdf5'

    model_checkpoint = ModelCheckpoint(model_file_name, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
    reduce_lr_on_plateu = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.3, verbose=1)
    plot_curves_callback = PlotLearningCurves()
    callbacks = [model_checkpoint, early_stopping, reduce_lr_on_plateu, plot_curves_callback]
    print('Start fitting...')
    model.fit(db[0], db[1], batch_size=batch_size, epochs=nb_epochs,
              validation_data=(db[2], db[3]),
              verbose=1, shuffle=True,
              callbacks=callbacks)

    print("Done!")


if __name__ == '__main__':
    train_model()