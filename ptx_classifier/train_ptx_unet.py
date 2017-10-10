import numpy as np
np.random.seed(1)
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc

from aid_funcs import image

from aid_funcs.keraswrapper import get_unet, load_model, plot_first_layer, PlotLearningCurves
from utils import *

train_data_lst, val_data_lst = train_val_partition()


def prep_data():
    nb_train = len(train_data_lst)
    nb_val = len(val_data_lst)
    train_imgs_arr = np.zeros((nb_train, 1, im_size, im_size))
    train_masks_arr = np.zeros((nb_train, 1, im_size, im_size), dtype=np.uint8)
    val_imgs_arr = np.zeros((nb_val, 1, im_size, im_size))
    val_masks_arr = np.zeros((nb_val, 1, im_size, im_size), dtype=np.uint8)
    for i in range(nb_train):
        train_imgs_arr[i] = image.imresize(train_data_lst[i]['img'], (im_size, im_size))
        if train_data_lst[i]['ptx_mask'] is None:
            ptx_mask = np.zeros((im_size, im_size), dtype=np.uint8)
        else:
            ptx_mask = image.imresize(train_data_lst[i]['ptx_mask'], (im_size, im_size))
        ptx_mask[ptx_mask > 0] = 1
        train_masks_arr[i] = ptx_mask
    for i in range(nb_val):
        val_imgs_arr[i] = image.imresize(val_data_lst[i]['img'], (im_size, im_size))
        if val_data_lst[i]['ptx_mask'] is None:
            ptx_mask = np.zeros((im_size, im_size), dtype=np.uint8)
        else:
            ptx_mask = image.imresize(val_data_lst[i]['ptx_mask'], (im_size, im_size))
        ptx_mask[ptx_mask > 0] = 1
        val_masks_arr[i] = ptx_mask
    db = (train_imgs_arr, train_masks_arr, val_imgs_arr, val_masks_arr)
    return db

def get_lung_masks(data_lst):
    nb_images = len(data_lst)
    lung_masks_arr = np.zeros((nb_images, 1, im_size, im_size), dtype=np.uint8)
    for i, case in enumerate(data_lst):
        lung_masks_arr[i] = image.imresize(case['lung_mask'], (im_size, im_size))
    lung_masks_arr[lung_masks_arr > 0] = 1
    return lung_masks_arr

def augment_data(db):
    pass

def analyze_performance(db):
    model_path = 'ptx_model_unet' + '.hdf5'
    model = load_model(model_path, custom_objects='dice_coef_loss')
    lung_masks_arr = get_lung_masks(val_data_lst)
    ptx_pred = model.predict(db[2], batch_size=10, verbose=1)
    ptx_pred *= lung_masks_arr
    # calculating ROC per pixel
    fpr, tpr, thresh = roc_curve(db[3].flatten(), ptx_pred.flatten())
    roc_auc = auc(fpr, tpr)
    dist_to_opt = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
    opt_ind = np.argmin(dist_to_opt)
    opt_thresh = thresh[opt_ind]

    # plotting the roc
    plt.figure(1)
    plt.plot(fpr, tpr, label='ROC')
    # plt.plot(fpr, thresh, label='Threshold')
    plt.plot(fpr[opt_ind], tpr[opt_ind], 'ro', label='Optimal thresh')
    plt.minorticks_on()
    plt.grid(b=True, which='both')
    plt.legend(loc='upper right')
    plt.title('ROC curve (area = %0.2f, opt thresh = %0.2f)' % (100 * roc_auc, opt_thresh))
    plt.savefig('roc analysis unet.png')

def train_model(db):
    lr = 0.0001
    optim_fun = Adam(lr=lr)

    model = get_unet(im_size, filters=32, optim_fun=optim_fun)
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


if __name__ == '__main__':
    db = prep_data()
    model = train_model(db)
    analyze_performance(db)
