import numpy as np
from skimage import measure
from collections import namedtuple
from multiprocessing import Pool

np.random.seed(1)
import os
import matplotlib.pyplot as plt
from aid_funcs.plot import show_image_with_overlay
from aid_funcs.misc import zip_load
from aid_funcs import image
from ptx_classifier.training_path import training_path
model_path = r'ptx_model_13_38_30_09_2017.hdf5'

im_size = 512
patch_sz = 32
smooth = 1.
max_num_of_patches = 4000000
pertubation_range = (-30, 30)
nb_augmentation = 9

Case = namedtuple('Case', ['name', 'img', 'lung_mask', 'ptx_mask'])

def display_train_set():
    train_set = zip_load(os.path.join(training_path, 'train_set.pkl'))
    for i, case in enumerate(train_set):
        show_image_with_overlay(case.img, case.lung_mask, case.ptx_mask, case.name + ' ' + str(i))
        plt.pause(1e-2)
        plt.waitforbuttonpress()


def is_ptx_case(ptx_mask):
    if ptx_mask is None or np.sum(ptx_mask) == 0:
        return False
    else:
        return True


def process_and_augment_data():
    train_data_lst, val_data_lst = train_val_partition()
    processed_train_data_lst = []
    processed_val_data_lst = []
    for case in train_data_lst:
        processed_train_data_lst += process_case(case, 0)
        if len(processed_train_data_lst) % 100 == 0:
            print('finished processing {} training samples (augmented)'.format(len(processed_train_data_lst)))
    for case in val_data_lst:
        processed_val_data_lst.append(process_case(case, 1))
        if len(processed_val_data_lst) % 10 == 0:
            print('finished processing {} validation samples'.format(len(processed_val_data_lst)))

    # with Pool(1) as pool:
    #     processed_train_data_lst = pool.starmap(process_case, [train_data_lst, 0])
    #     processed_val_data_lst = pool.starmap(process_case, [val_data_lst, 1])
    print('total of {} training samples processed (augmented) and {} validation'.format(len(processed_train_data_lst),
                                                                                        len(processed_val_data_lst)))
    return processed_train_data_lst, processed_val_data_lst


def load_data_lst():
    data_lst = zip_load(os.path.join(training_path, 'train_set.pkl'))
    return data_lst


def process_case(case, set):
    pertubation_vec = np.random.randint(*pertubation_range, size=(nb_augmentation, 4))
    bb = get_lung_bb(case.lung_mask)
    img, lung_mask, ptx_mask = crop_n_resize(case, bb)
    img = normalize_img(img)
    out_lst = [Case(case.name, img, lung_mask, ptx_mask)]
    if set == 0:
        for i in range(nb_augmentation):
            pert_bb = np.array(bb) + np.array(pertubation_vec[i])
            img, lung_mask, ptx_mask = crop_n_resize(case, pert_bb)
            img = normalize_img(img)
            out_lst.append(Case(case.name, img, lung_mask, ptx_mask))
    else:
        out_lst = out_lst[0]
    # import matplotlib.pyplot as plt
    # for i in range(len(out_lst)):
    #     plt.subplot(3,4,i+1)
    #     plt.imshow(out_lst[i].img, cmap='gray')
    # plt.show()
    return out_lst


def get_lung_bb(lung_mask):
    lung_map_dilated = image.safe_binary_morphology(lung_mask, sesize=15, mode='dilate')
    lung_bbox = measure.regionprops(lung_map_dilated.astype(np.uint8))
    return lung_bbox[0].bbox


def normalize_img(img):
    # Normalizing each image based on mean and std of lung pixels
    img = img.astype(np.float32)
    mean_val = np.nanmean(img)
    std_val = np.nanstd(img)
    out_img = img.copy()
    out_img -= mean_val
    out_img /= std_val
    out_img[np.isnan(out_img)] = np.nanmax(out_img)
    return out_img


def crop_n_resize(case, bb):
    bb = np.clip(bb, 0, case.img.shape[0])
    img = case.img[bb[0]:bb[2], bb[1]:bb[3]]
    img = image.imresize(img.astype(np.float32), (im_size, im_size) )

    lung_mask = case.lung_mask[bb[0]:bb[2], bb[1]:bb[3]]
    lung_mask = image.imresize(lung_mask, (im_size, im_size))

    if case.ptx_mask is not None:
        ptx_mask = case.ptx_mask[bb[0]:bb[2], bb[1]:bb[3]]
        ptx_mask = image.imresize(ptx_mask, (im_size, im_size))
    else:
        ptx_mask = None

    return img, lung_mask, ptx_mask


def train_val_partition(data_lst=None):
    if data_lst is None:
        data_lst = load_data_lst()
    nb_train_total = len(data_lst)
    val_idx = np.random.choice(range(nb_train_total), int(0.3 * nb_train_total))

    # Partition to train and val sets
    n_val = len(val_idx)
    n_train = nb_train_total - n_val
    print('Partition to validation (n={}) and training (n={}) sets'.format(n_val, n_train))
    val_data_lst = []
    train_data_lst = []
    for i in range(nb_train_total):
        if i in val_idx:
            val_data_lst.append(data_lst[i])
        else:
            train_data_lst.append(data_lst[i])
    return train_data_lst, val_data_lst