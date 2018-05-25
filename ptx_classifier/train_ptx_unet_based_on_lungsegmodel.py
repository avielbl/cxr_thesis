import shutil
import time

import Augmentor
import cv2
from keras import Model, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv2D
from keras.optimizers import Adam

from CXRLoadNPrep import load_dicom
from aid_funcs.misc import load_from_h5
from aid_funcs.keraswrapper import load_model, PlotLearningCurves, dice_coef_loss, dice_coef
from ptx_classifier.utils import *

# Setting folders
current_path = r"C:\projects\CXR_thesis\ptx_classifier"
train_images_path = os.path.join(current_path, "train_images")
train_masks_path = os.path.join(current_path, "train_masks")
val_images_path = os.path.join(current_path, "val_images")
val_masks_path = os.path.join(current_path, "val_masks")
augmented_images_path = os.path.join(train_images_path, 'augmented')
augmented_masks_path = os.path.join(train_masks_path, 'augmented')


def get_val_idx(image_list):
    nb_train_total = len(image_list)
    return np.random.choice(range(nb_train_total), int(0.3 * nb_train_total), replace=False)


def load_and_prep_data():
    """
    function for loading all training set orig dicom and masks, separate to train and validation and save png files
    to be used further for augmentation and preprocessing.
    Saved masks are 0 for background, 1 for lungs, 2 for ptx
    :return: None
    """
    # Gather both positive and negative cases
    pos_path = os.path.join(training_path, 'pos_cases')
    neg_path = os.path.join(training_path, 'neg_cases')
    pos_files = [os.path.join(pos_path, file) for file in os.listdir(pos_path)]
    neg_files = [os.path.join(neg_path, file) for file in os.listdir(neg_path)]
    imgs_path_lst = pos_files + neg_files
    lung_seg_path = os.path.join(training_path, 'lung_seg_gt')
    ptx_masks_path = os.path.join(training_path, 'ptx_masks_gt')

    # Randomly generate validation set
    val_idx = get_val_idx(imgs_path_lst)

    for im_count, curr_img_path in enumerate(imgs_path_lst):
        # Iterate through all set
        img_name = os.path.split(curr_img_path)[1][:-4]

        # read orig dicom and resize to square 1024 pixels
        img = load_dicom(curr_img_path)
        img = image.imresize(img, (1024, 1024))
        mask = np.zeros((1024, 1024))
        lung_path = os.path.join(lung_seg_path, img_name + '.png')

        if os.path.isfile(lung_path):
            # Loading lung mask
            lung_mask = cv2.imread(lung_path, cv2.IMREAD_GRAYSCALE)
            lung_mask = image.imresize(lung_mask, img.shape)
            mask[lung_mask > 0] = 1
            # Loading ptx mask if exist
            ptx_path = os.path.join(ptx_masks_path, img_name + '.png')
            if os.path.isfile(ptx_path):
                ptx_mask = cv2.imread(ptx_path, cv2.IMREAD_GRAYSCALE)
                ptx_mask = image.imresize(ptx_mask, img.shape)
                ptx_mask[lung_mask == 0] = 0
            else:
                ptx_mask = np.zeros((1024, 1024))
            mask[ptx_mask > 0] = 2
            if im_count in val_idx:
                out_images = val_images_path
                out_masks = val_masks_path
            else:
                out_images = train_images_path
                out_masks = train_masks_path
            cv2.imwrite(os.path.join(out_images, img_name + '.png'), img)
            cv2.imwrite(os.path.join(out_masks, img_name + '.png'), mask)
        else:
            print("couldn't find lung mask for image {}".format(img_name))
        print("Loaded image number %i" % im_count)


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
        # Loading image and mask
        image = cv2.imread(os.path.join(images_path, images_list[i]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(masks_path, masks_list[i]), cv2.IMREAD_GRAYSCALE)
        # Cropping around lungs and resize
        image, mask = get_lung_bb(image, mask)
        image = image.imresize(image, (im_size, im_size))
        mask = image.imresize(mask, (im_size, im_size))

        # Normalizing intensity based on lung pixels only
        nan_img = image.copy()
        nan_img = nan_img.astype(np.float32)
        nan_img[mask == 0] = np.nan
        mean_val = np.nanmean(nan_img)
        std_val = np.nanstd(nan_img)
        out_img = image.copy().astype(np.float32)
        out_img -= mean_val
        out_img /= std_val

        images_arr[i, :, :, 0] = out_img
        mask[mask > 0] = 1
        masks_arr[i, :, :, 0] = mask
    return images_arr, masks_arr


def get_lung_bb(img, mask):
    lung_bbox = measure.regionprops(mask.astype(np.uint8))
    lung_bbox = lung_bbox[0].bbox
    img = img[lung_bbox[0]:lung_bbox[2], lung_bbox[1]:lung_bbox[3]]
    mask = mask[lung_bbox[0]:lung_bbox[2], lung_bbox[1]:lung_bbox[3]]
    return img, mask


def load_old_data():
    print('Loading data...')
    db = [
        load_from_h5(os.path.join(training_path, 'db_train_imgs_arr.h5')),
        load_from_h5(os.path.join(training_path, 'db_train_masks_arr.h5')).astype(np.uint8),
        load_from_h5(os.path.join(training_path, 'db_val_imgs_arr.h5')),
        load_from_h5(os.path.join(training_path, 'db_val_masks_arr.h5')).astype(np.uint8)
    ]

    # switch from channels first to channels last
    db[0] = np.rollaxis(db[0], 1, 4)
    db[1] = np.rollaxis(db[1], 1, 4)
    db[2] = np.rollaxis(db[2], 1, 4)
    db[3] = np.rollaxis(db[3], 1, 4)
    return db


def create_model():
    base_model = load_model(r"C:\projects\CXR_thesis\new_lung_segmentation\lung_seg_model_12_54_20_05_2018.hdf5",
                            custom_objects='dice_coef_loss')
    # new_inputs_layer = Input((512, 512, 1))
    # base_model.layers[0] = new_inputs_layer
    x = base_model.get_layer('dropout_2').output
    prediction = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal', name='prediction')(x)

    return Model(inputs=base_model.input, outputs=prediction)


def train_model(model, db):
    nb_epochs = 100
    batch_size = 2
    lr = 0.0001
    model.compile(optimizer=Adam(lr=lr), loss=dice_coef_loss, metrics=[dice_coef])

    time_strftime = time.strftime("%H_%M_%d_%m_%Y")
    model_file_name = 'ptx_unet_based_on_lungsegmodel_' + time_strftime + '.hdf5'
    model_checkpoint = ModelCheckpoint(model_file_name, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
    reduce_lr_on_plateu = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.3, verbose=1)
    plot_curves_callback = PlotLearningCurves(metric_name='dice_coef')

    model.fit(db[0], db[1], batch_size=batch_size, epochs=nb_epochs,
            verbose=1, shuffle=True, validation_data=(db[2], db[3]),
            callbacks=[model_checkpoint, early_stopping, reduce_lr_on_plateu, plot_curves_callback])


def main():
    load_and_prep_data()
    augment_data()
    train_images_arr, train_masks_arr = pre_process_data(augmented_images_path, augmented_masks_path)
    val_images_arr, val_masks_arr = pre_process_data(val_images_path, val_masks_path)
    # db = load_old_data()
    model = create_model()
    train_model(model, [train_images_arr, train_masks_arr, val_images_arr, val_masks_arr])


if __name__ == '__main__':
    main()