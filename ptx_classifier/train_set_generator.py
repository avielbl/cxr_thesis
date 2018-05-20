import numpy as np
np.random.seed(1)
from aid_funcs.misc import zip_save
from scipy.misc import imread
from aid_funcs import CXRLoadNPrep as clp
from ptx_classifier.utils import *

# Creating list of images
pos_path = os.path.join(training_path, 'pos_cases')
neg_path = os.path.join(training_path, 'neg_cases')
pos_files = [os.path.join(pos_path, file) for file in os.listdir(pos_path)]
neg_files = [os.path.join(neg_path, file) for file in os.listdir(neg_path)]
imgs_path_lst = pos_files + neg_files
nb_train_imgs = len(imgs_path_lst)

lung_seg_path = os.path.join(training_path, 'lung_seg_gt')
ptx_masks_path = os.path.join(training_path, 'ptx_masks_gt')
train_set_lst = []

for im_count, curr_img_path in enumerate(imgs_path_lst):
    img_name = os.path.split(curr_img_path)[1][:-4]
    img = clp.load_dicom(curr_img_path)
    img = image.square_image(img)
    img = image.imresize(img, (1024, 1024))
    lung_path = os.path.join(lung_seg_path, img_name + '.png')
    if os.path.isfile(lung_path):
        # Loading lung mask
        lung_mask = imread(lung_path, mode='L')
        lung_mask = image.imresize(lung_mask, img.shape)

        # Loading ptx mask if exist
        ptx_path = os.path.join(ptx_masks_path, img_name + '.png')
        if os.path.isfile(ptx_path):
            ptx_mask = imread(ptx_path, mode='L')
            ptx_mask = image.imresize(ptx_mask, img.shape)
            ptx_mask[lung_mask == 0] = 0
        else:
            ptx_mask = None

        case = Case(img_name, img, lung_mask, ptx_mask)
        train_set_lst.append(case)
    print("Loaded image number %i" % im_count)
# Creating list of images
pos_path = os.path.join(training_path, 'pos_cases')
neg_path = os.path.join(training_path, 'neg_cases')
pos_files = [os.path.join(pos_path, file) for file in os.listdir(pos_path)]
neg_files = [os.path.join(neg_path, file) for file in os.listdir(neg_path)]
imgs_path_lst = pos_files + neg_files
nb_train_imgs = len(imgs_path_lst)

lung_seg_path = os.path.join(training_path, 'lung_seg_gt')
ptx_masks_path = os.path.join(training_path, 'ptx_masks_gt')
train_set_lst = []

for im_count, curr_img_path in enumerate(imgs_path_lst):
    img_name = os.path.split(curr_img_path)[1][:-4]
    img = clp.load_dicom(curr_img_path)
    img = image.square_image(img)
    img = image.imresize(img, (1024, 1024))
    lung_path = os.path.join(lung_seg_path, img_name + '.png')
    if os.path.isfile(lung_path):
        # Loading lung mask
        lung_mask = imread(lung_path, mode='L')
        lung_mask = image.imresize(lung_mask, img.shape)

        # Loading ptx mask if exist
        ptx_path = os.path.join(ptx_masks_path, img_name + '.png')
        if os.path.isfile(ptx_path):
            ptx_mask = imread(ptx_path, mode='L')
            ptx_mask = image.imresize(ptx_mask, img.shape)
            ptx_mask[lung_mask == 0] = 0
        else:
            ptx_mask = None

        case = Case(img_name, img, lung_mask, ptx_mask)
        train_set_lst.append(case)
    print("Loaded image number %i" % im_count)


zip_save(train_set_lst, os.path.join(training_path, 'train_set.pkl'))
