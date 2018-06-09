import shutil
import time

import Augmentor
import cv2
from keras import Model, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv2D
from keras.optimizers import Adam

from CXRLoadNPrep import load_dicom
from aid_funcs.misc import load_from_h5, printProgressBar, roc_plotter
from aid_funcs.keraswrapper import load_model, PlotLearningCurves, dice_coef_loss, dice_coef, get_class_weights, \
    weighted_pixelwise_crossentropy, focal_loss
from image import imresize, im_rescale
from ptx_classifier.utils import *

# Setting folders
current_path = os.path.split(__file__)[0]
train_images_path = os.path.join(current_path, "train_images")
train_masks_path = os.path.join(current_path, "train_masks")
val_images_path = os.path.join(current_path, "val_images")
val_masks_path = os.path.join(current_path, "val_masks")
augmented_train_images_path = os.path.join(train_images_path, 'augmented')
augmented_train_masks_path = os.path.join(train_masks_path, 'augmented')
augmented_val_images_path = os.path.join(val_images_path, 'augmented')
augmented_val_masks_path = os.path.join(val_masks_path, 'augmented')


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
        img = im_rescale(img, 0, 255)
        mask = np.zeros((1024, 1024))
        lung_path = os.path.join(lung_seg_path, img_name + '.png')

        if os.path.isfile(lung_path):
            # Loading lung mask
            lung_mask = cv2.imread(lung_path, cv2.IMREAD_GRAYSCALE)
            lung_mask = image.imresize(lung_mask, img.shape)
            mask[lung_mask > 0] = 127
            # Loading ptx mask if exist
            ptx_path = os.path.join(ptx_masks_path, img_name + '.png')
            if os.path.isfile(ptx_path):
                ptx_mask = cv2.imread(ptx_path, cv2.IMREAD_GRAYSCALE)
                ptx_mask = image.imresize(ptx_mask, img.shape)
                ptx_mask[lung_mask == 0] = 0
            else:
                ptx_mask = np.zeros((1024, 1024))
            mask[ptx_mask > 0] = 255
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


def augment_data(images_in_path, masks_in_path, images_out_path, masks_out_path):
    p = Augmentor.Pipeline(images_in_path, output_directory=images_out_path)
    nb = len(os.listdir(images_in_path))
    p.ground_truth(masks_in_path)
    p.rotate(probability=0.3, max_left_rotation=5, max_right_rotation=5)
    p.zoom(probability=0.3, min_factor=0.8, max_factor=1.2)
    p.random_distortion(probability=0.9, grid_width=4, grid_height=4, magnitude=4)
    p.sample(nb * 10)
    images_list = os.listdir(images_out_path)
    masks_list = [file for file in images_list if 'groundtruth' in file]
    for mask in masks_list:
        shutil.move(os.path.join(images_out_path, mask), os.path.join(masks_out_path, mask))


def get_lung_masks(masks_path):
    masks_list = os.listdir(masks_path)
    nb_masks = len(masks_list)
    masks_arr = np.zeros((nb_masks, im_size, im_size, 1), np.uint8)
    for i in range(nb_masks):
        mask = cv2.imread(os.path.join(masks_path, masks_list[i]), -1)
        lung_mask = np.uint8(mask == 127)
        masks_arr[i, :, :, 0] = lung_mask

    return masks_arr


def pre_process_data(images_path, masks_path):
    images_list = os.listdir(images_path)
    masks_list = os.listdir(masks_path)
    nb_images = len(images_list)
    images_arr = np.zeros((nb_images, im_size, im_size, 1), np.float32)
    masks_arr = np.zeros((nb_images, im_size, im_size, 1), np.uint8)
    lung_masks_arr = np.zeros((nb_images, im_size, im_size, 1), np.uint8)
    print('Pre-processing data from: {}'.format(images_path))
    printProgressBar(0, nb_images, prefix='Progress:', suffix='Complete', bar_length=50)

    for i in range(nb_images):
        # Loading image and mask
        image = cv2.imread(os.path.join(images_path, images_list[i]), -1)
        mask = cv2.imread(os.path.join(masks_path, masks_list[i]), -1)
        # Cropping around lungs and resize
        image, mask = get_lung_bb(image, mask)
        image = imresize(image, (im_size, im_size))
        mask = imresize(mask, (im_size, im_size))

        # Normalizing intensity based on lung pixels only
        nan_img = image.copy()
        nan_img = nan_img.astype(np.float32)
        nan_img[mask == 0] = np.nan
        mean_val = np.nanmean(nan_img)
        std_val = np.nanstd(nan_img)
        if std_val < 1e-2:
            print('std small')
        out_img = image.copy().astype(np.float32)
        out_img -= mean_val
        out_img /= std_val

        images_arr[i, :, :, 0] = out_img
        lung_mask = np.uint8(mask > 0)
        mask[mask < 255] = 0
        mask[mask > 0] = 1
        masks_arr[i, :, :, 0] = mask
        lung_masks_arr[i, :, :, 0] = lung_mask
        printProgressBar(i + 1, nb_images, prefix='Progress:', suffix='Complete', bar_length=50)

    return images_arr, masks_arr, lung_masks_arr


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


def create_model(labels):
    print('Loading lung seg model..')
    base_model = load_model(r"C:\Users\admin\PycharmProjects\CXR_thesis\new_lung_segmentation\lung_seg_model.hdf5",
                            custom_objects='dice_coef_loss')
    # new_inputs_layer = Input((512, 512, 1))
    # base_model.layers[0] = new_inputs_layer
    x = base_model.get_layer('dropout_2').output
    prediction = Conv2D(labels, (1, 1), activation='softmax', kernel_initializer='he_normal', name='prediction')(x)

    return Model(inputs=base_model.input, outputs=prediction)


def train_model_focal_loss(db):
    model = create_model(1)

    nb_epochs = 100
    batch_size = 5
    lr = 0.0001

    # Pretraining
    model.compile(optimizer=Adam(lr=.001), loss='binary_crossentropy', metrics=[dice_coef])
    model.fit(db[0], db[1], batch_size=batch_size, epochs=2,
              verbose=1, shuffle=True, validation_data=(db[2], db[3]))

    # Defining callbacks
    time_strftime = time.strftime("%H_%M_%d_%m_%Y")
    model_file_name = 'ptx_unet_based_on_lungsegmodel_FL_' + time_strftime + '.hdf5'
    model_checkpoint = ModelCheckpoint(model_file_name, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
    reduce_lr_on_plateu = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.3, verbose=1)
    plot_curves_callback = PlotLearningCurves(metric_name='dice_coef')

    model.compile(optimizer=Adam(lr=lr), loss=[focal_loss()], metrics=[dice_coef])
    model.fit(db[0], db[1], batch_size=batch_size, epochs=nb_epochs,
              verbose=1, shuffle=True, validation_data=(db[2], db[3]),
              callbacks=[model_checkpoint, early_stopping, reduce_lr_on_plateu, plot_curves_callback])


def train_model_WCE(db):
    model = create_model(2)
    nb_epochs = 100
    batch_size = 5
    lr = 0.0001
    print('Calculating class weights..')
    # class_weights = get_class_weights(db[1])
    class_weights = [1., 10.]
    db[1] = categorize(db[1])
    db[3] = categorize(db[3])

    time_strftime = time.strftime("%H_%M_%d_%m_%Y")
    model_file_name = 'ptx_unet_based_on_lungsegmodel_WCE_' + time_strftime + '.hdf5'
    model_checkpoint = ModelCheckpoint(model_file_name, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
    reduce_lr_on_plateu = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.3, verbose=1)
    plot_curves_callback = PlotLearningCurves(metric_name='dice_coef')

    model.compile(optimizer=Adam(lr=lr), loss=[weighted_pixelwise_crossentropy(class_weights)], metrics=[dice_coef])
    model.fit(db[0], db[1], batch_size=batch_size, epochs=nb_epochs,
              verbose=1, shuffle=True, validation_data=(db[2], db[3]),
              callbacks=[model_checkpoint, early_stopping, reduce_lr_on_plateu, plot_curves_callback])
    return model


def categorize(arr):
    out = np.zeros((arr.shape[0], arr.shape[1], arr.shape[2], 2), dtype=np.uint8)
    out[:, :, :, 0] = (1 - arr).squeeze()
    out[:, :, :, 1] = arr.squeeze()
    return out


def validate_performance(model_name, model, val_images_arr, val_masks_arr, lung_masks_arr):
    print('Validating model {}'.format(model_name))
    scores = model.predict(val_images_arr, batch_size=5, verbose=1)
    scores = scores[:, :, :, 1]  # Taking only scores for ptx
    # calculating ROC per pixel
    lung_flatten = np.array(lung_masks_arr).flatten()
    lung_idx = np.where(lung_flatten > 0)
    scores_vec = scores.flatten()
    scores_vec = scores_vec[lung_idx]
    gt_vec = np.uint8(val_masks_arr.flatten())
    gt_vec = gt_vec[lung_idx]
    roc_plotter(gt_vec, scores_vec, model_name)

    pred_maps = np.zeros_like(scores, dtype=np.uint8)
    pred_maps[scores > 0.5] = 255

    nb_val = val_images_arr.shape[0]
    plt.figure()
    if os.path.isdir('results') is False:
        os.mkdir('results')
    for i in range(nb_val):
        img = np.squeeze(val_images_arr[i])
        pred = np.squeeze(pred_maps[i]) * np.squeeze(lung_masks_arr[i])

        gt = np.squeeze(val_masks_arr[i])

        show_image_with_overlay(img, overlay=gt, overlay2=pred, alpha=0.8)
        plt.axis('off')
        out_path = 'results\\' + str(i) + '.png'
        plt.savefig(out_path, bbox_inches='tight')
        print('saved image {}/{}'.format(i, nb_val))


def main():
    # load_and_prep_data()
    # augment_data(train_images_path, train_masks_path, augmented_train_images_path, augmented_train_masks_path)
    # augment_data(val_images_path, val_masks_path, augmented_val_images_path, augmented_val_masks_path)
    print('Pre-processing training set..')
    train_images_arr, train_masks_arr, _ = pre_process_data(augmented_train_images_path, augmented_train_masks_path)
    print('Pre-processing validation set..')
    val_images_arr, val_masks_arr, val_lung_masks_arr = pre_process_data(augmented_val_images_path, augmented_val_masks_path)
    # db = load_old_data()
    # train_model_focal_loss([train_images_arr, train_masks_arr, val_images_arr, val_masks_arr])
    # validate_performance('ptx_unet_based_on_lungsegmodel_FL_22_45_07_06_2018.hdf5',
    #                      {'loss': [focal_loss()], 'dice_coef': dice_coef})

    model = train_model_WCE([train_images_arr, train_masks_arr, val_images_arr, val_masks_arr])
    validate_performance('ptx_unet_based_on_lungsegmodel_WCE', model, val_images_arr, val_masks_arr, val_lung_masks_arr)


if __name__ == '__main__':
    main()
