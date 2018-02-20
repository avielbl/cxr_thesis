import numpy as np
import matplotlib.pyplot as plt
import CXRLoadNPrep as clp
from aid_funcs import image
import os
import cv2
import keras.models
import pickle
import time
# from drawnow import drawnow
from sklearn.metrics import roc_curve, auc
from scipy import ndimage
from skimage import morphology

testing_path = 'C:\\Projects\\Algorithm_Dev\\CXR\\DATA\\Testing_Data\\all-dicom'
lung_seg_path = 'C:\Projects\Algorithm_Dev\CXR\DATA\Testing_Data\segmentation_maps'
im_size = 512


def load_lungs_maps():
    seg_map_dir = os.listdir(lung_seg_path)
    n = len(seg_map_dir)
    seg_map_arr = np.ndarray((n, 1, im_size, im_size), dtype='uint16')
    im_count = 0
    print("-" * 30)
    for image_name in seg_map_dir:
        seg_map = cv2.imread(os.path.join(lung_seg_path, image_name), cv2.IMREAD_GRAYSCALE)
        seg_map = np.uint8(image.im_rescale(seg_map))
        seg_map = image.resize_w_aspect(seg_map, im_size)
        seg_map = np.uint8(seg_map)
        seg_map_arr[im_count] = seg_map
        im_count += 1
        print("Loaded lungs map number %i" % im_count)
    return seg_map_arr


def load_testing_images():
    pre_process_params_path = "C:\\Projects\\Algorithm_Dev\\CXR\\new_ptx_classifier\\pre_process_params.pickle"
    with open(pre_process_params_path, 'rb') as f:
        mean_val, std_val = pickle.load(f)
    images_dir = os.listdir(testing_path)
    n = len(images_dir)
    images_arr = np.ndarray((n, 1, 1, im_size, im_size), dtype='float32')
    lung_seg_map_arr = load_lungs_maps()
    im_count = 0
    print("-" * 30)
    for image_name in images_dir:
        img = clp.load_dicom(os.path.join(testing_path, image_name))
        img = image.im_rescale(img, 0, 2 ** 16)
        img = image.resize_w_aspect(img, im_size)
        lung_map = lung_seg_map_arr[im_count]
        lung_map = lung_map.squeeze()
        img[lung_map == 0] = np.nan
        images_arr[im_count] = img
        im_count += 1
        print("Loaded image number %i" % im_count)
    images_arr -= mean_val
    images_arr /= std_val
    images_arr[np.isnan(images_arr)] = mean_val / std_val
    return images_arr


def load_test_ptx_maps():
    ptx_maps_path = "C:\\Projects\\Algorithm_Dev\\CXR\\DATA\\Testing_Data\\ptx_maps"
    ptx_map_dir = os.listdir(ptx_maps_path)
    n = len(ptx_map_dir)
    ptx_map_arr = np.ndarray((n, im_size, im_size), dtype='uint16')
    im_count = 0
    print("-" * 30)
    for image_name in ptx_map_dir:
        ptx_map = cv2.imread(os.path.join(ptx_maps_path, image_name), cv2.IMREAD_GRAYSCALE)
        ptx_map = np.uint8(image.im_rescale(ptx_map))
        ptx_map = image.resize_w_aspect(ptx_map, im_size)
        ptx_map = np.uint8(ptx_map)
        ptx_map_arr[im_count] = ptx_map
        im_count += 1
        print("Loaded segmentation image number %i" % im_count)
    return ptx_map_arr


def predict_all_masks():
    results_folder_path = 'C:\Projects\Algorithm_Dev\CXR\\new_ptx_classifier'
    model = keras.models.load_model("C:\Projects\Algorithm_Dev\CXR\\new_ptx_classifier\ptx_net_080117.hdf5")
    images_arr = load_testing_images()
    gt_ptx_maps = load_test_ptx_maps()
    n = len(images_arr)
    ptx_masks_arr = np.ndarray((n, im_size, im_size), dtype='uint16')
    scores_arr = np.ndarray((n, im_size, im_size))

    # actual prediction
    for i in range(n):
        start_time = time.time()
        img = images_arr[i]
        scores = model.predict(img, verbose=0)
        scores = np.squeeze(scores)
        scores_arr[i] = scores
        print('predicted %i/%i in %f.3 seconds' % (i + 1, n, time.time() - start_time))

    # calculating ROC per pixel
    fpr, tpr, thresh = roc_curve(gt_ptx_maps.flatten(), scores_arr.flatten())
    roc_auc = auc(fpr, tpr)
    dist_to_opt = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
    opt_ind = np.argmin(dist_to_opt)
    opt_thresh = thresh[opt_ind]

    # plotting the roc
    plt.figure(1)
    plt.plot(fpr, tpr, label='ROC')
    plt.plot(fpr, thresh, label='Threshold')
    plt.plot(fpr[opt_ind], tpr[opt_ind], 'ro', label='Optimal thresh')
    plt.minorticks_on()
    plt.grid(b=True, which='both')
    plt.legend(loc='upper right')
    plt.title('ROC curve (area = %0.2f, opt thresh = %0.2f)' % (100 * roc_auc, opt_thresh))
    plt.savefig(os.path.join(results_folder_path, 'roc analysis.png'))

    #  post-processing
    for i in range(n):
        start_time = time.time()
        scores = np.squeeze(scores_arr[i])
        predicted_mask = np.zeros_like(scores)
        predicted_mask[scores >= 0.11] = 1
        label_im, nb_labels = ndimage.label(predicted_mask)
        # remove small objects
        if nb_labels > 2:
            areas = ndimage.sum(predicted_mask, label_im, range(nb_labels + 1))
            sorted_areas = np.sort(areas)
            smallest_lung_ares = sorted_areas[-2]
            mask_size = areas < smallest_lung_ares
            remove_pixel = mask_size[label_im]
            label_im[remove_pixel] = 0
            lungs_only_mask = np.zeros_like(label_im)
            lungs_only_mask[label_im > 0] = 1
        else:
            lungs_only_mask = predicted_mask
        # closing gaps along lungs contour and fill holes
        pp_lungs_mask = np.zeros_like(label_im, bool)
        labels = np.unique(label_im)
        labels = labels[1:]
        for lung in labels:
            close_size = 30
            curr_lung = np.zeros_like(label_im, bool)
            curr_lung[label_im == lung] = True
            se = morphology.disk(close_size)
            pad_width = ((close_size, close_size), (close_size, close_size))
            padded_mask = np.pad(curr_lung, pad_width, mode='constant')
            curr_lung = morphology.binary_closing(padded_mask, se)
            curr_lung = curr_lung[close_size:-close_size, close_size:-close_size]
            curr_lung = morphology.remove_small_holes(curr_lung, im_size ** 2 / 3)
            pp_lungs_mask[curr_lung] = True
        ptx_masks_arr[i] = pp_lungs_mask.astype('uint16')
        print('completed post-process of %i/%i in %.3f seconds' % (i + 1, n, time.time() - start_time))

    # performance analysis
    sens = np.ndarray(n)
    sens.fill(np.nan)
    spec = np.ndarray(n)
    spec.fill(np.nan)
    acc = np.ndarray(n)
    acc.fill(np.nan)
    omega = np.ndarray(n)
    omega.fill(np.nan)
    dice = np.ndarray(n)
    dice.fill(np.nan)

    for i in range(n):
        gt_mask = np.squeeze(gt_ptx_maps[i])
        gt_mask = image.resize_w_aspect(gt_mask, im_size)
        gt_mask = gt_mask.astype('uint16')
        curr_mask = np.squeeze(ptx_masks_arr[i])
        tp = np.sum(gt_mask * curr_mask)
        tn = np.sum((1 - gt_mask) * (1 - curr_mask))
        fp = np.sum((1 - gt_mask) * curr_mask)
        fn = np.sum(gt_mask * (1 - curr_mask))
        sens[i] = 100 * tp / (tp + fn)
        spec[i] = 100 * tn / (tn + fp)
        acc[i] = 100 * (tp + tn) / (tp + tn + fp + fn)
        omega[i] = tp / (fp + tp + fn)
        dice[i] = 2 * tp / (2 * tp + fn + fp)
        im_name = 'pred_seg%02u' % (i + 1)
        img = np.squeeze(images_arr[i])
        plt.figure(i)
        image.show_image_with_overlay(img=img, overlay=256 * curr_mask, overlay2=256 * gt_mask,
                                      title_str=im_name + ': blue=gt, red=pred')
        results_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_results'))
        path = os.path.join(results_folder_path, '%s.png' % im_name)
        plt.savefig(path, bbox_inches='tight')

    results = {'mean_sens': np.nanmean(sens),
               'std_sens': np.nanstd(sens),
               'mean_spec': np.nanmean(spec),
               'std_spec': np.nanstd(spec),
               'mean_acc': np.nanmean(acc),
               'std_acc': np.nanstd(acc),
               'mean_omega': np.nanmean(omega),
               'std_omega': np.nanstd(omega),
               'mean_dice': np.nanmean(dice),
               'std_dice': np.nanstd(dice)}
    print('mean sensitivity: {} +- {}'.format(results['mean_sens'], results['std_sens']))
    print('mean specificity: {} +- {}'.format(results['mean_spec'], results['std_spec']))
    print('mean accuracy: {} +- {}'.format(results['mean_acc'], results['std_acc']))
    print('mean omega: {} +- {}'.format(results['mean_omega'], results['std_omega']))
    print('mean dice: {} +- {}'.format(results['mean_dice'], results['std_dice']))
    return (results, ptx_masks_arr)


if __name__ == '__main__':
    predict_all_masks()
