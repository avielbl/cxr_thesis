import numpy as np
import matplotlib.pyplot as plt
from aid_funcs import image
from aid_funcs.keraswrapper import load_model
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

model = load_model('ptx_model_unet.hdf5', custom_objects='dice_coef_loss')
db = prep_data()
scores = model.predict(db[2], batch_size=5, verbose=1)
pred_maps = np.zeros_like(scores, dtype=np.uint8)
pred_maps[scores > 0] = 255

nb_val = db[3].shape[0]
for i in range(nb_val):
    img = np.squeeze(db[2][i])
    pred = np.squeeze(pred_maps[i])
    gt = np.squeeze(db[3][i])
    show_image_with_overlay(img, overlay=gt, overlay2=pred)
    out_path = 'results\\' + str(i) + '.png'
    plt.savefig(out_path, bbox_inches='tight')
