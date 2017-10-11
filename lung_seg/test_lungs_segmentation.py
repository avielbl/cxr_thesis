from utilfuncs import *
import matplotlib.pyplot as plt
from aid_funcs import plot
import os
# import keras.models
from aid_funcs.keraswrapper import load_model
import time
import pickle
from drawnow import drawnow, figure
from predict import predict

testing_path = '..\\DATA\\Testing_Data\\all-dicom'
seg_maps_path = '..\\DATA\\Testing_Data\\segmentation_maps'

im_size = params.im_size

def predict_all_masks(result_iter):
    results_folder_path = '.\\test_results\\' + str(result_iter)
    if not os.path.isdir(results_folder_path):
        os.makedirs(results_folder_path)
    model = load_model(params.seg_model_path, custom_objects='dice_coef_loss')
    # model = keras.models.load_model(params.seg_model_path)
    images_arr = load_images(testing_path)
    images_arr = pre_process_images(images_arr)
    gt_seg_maps = load_segmentation_maps(seg_maps_path)
    n = len(images_arr)
    lung_masks_arr = np.ndarray((n, im_size, im_size), dtype='uint8')

    for i in range(n):
        start_time = time.time()
        curr_lung = predict(images_arr[i], model)
        lung_masks_arr[i] = curr_lung.r_lung_mask + curr_lung.l_lung_mask
        print('completed prediction of %i/%i in %.3f seconds' % (i + 1, n, time.time() - start_time))
        im_name = 'pred_seg%02u' % (i + 1)
        img = np.squeeze(images_arr[i])
        curr_mask = np.squeeze(lung_masks_arr[i])
        gt_mask = np.squeeze(gt_seg_maps[i])
        drawnow(plot.show_image_with_overlay, img=img, overlay=256 * curr_mask, overlay2=256 * gt_mask,
                title_str=im_name + ': blue=gt, red=pred')
        path = os.path.join(results_folder_path, '%s.png' % im_name)
        print('saving figure', i + 1)
        plt.savefig(path, bbox_inches='tight')
        plt.clf()

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
    figure()
    for i in range(n):
        gt_mask = np.squeeze(gt_seg_maps[i])
        curr_mask = np.squeeze(lung_masks_arr[i])
        tp = np.sum(gt_mask * curr_mask)
        tn = np.sum((1 - gt_mask) * (1 - curr_mask))
        fp = np.sum((1 - gt_mask) * curr_mask)
        fn = np.sum(gt_mask * (1 - curr_mask))
        sens[i] = 100 * tp / (tp + fn)
        spec[i] = 100 * tn / (tn + fp)
        acc[i] = 100 * (tp + tn) / (tp + tn + fp + fn)
        omega[i] = tp / (fp + tp + fn)
        dice[i] = 2 * tp / (2 * tp + fn + fp)
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
    print('mean sensitivity: %f +- %f' % (results['mean_sens'], results['std_sens']))
    print('mean specificity: %f +- %f' % (results['mean_spec'], results['std_spec']))
    print('mean accuracy: %f +- %f' % (results['mean_acc'], results['std_acc']))
    print('mean omega: %f +- %f' % (results['mean_omega'], results['std_omega']))
    print('mean dice: %f +- %f' % (results['mean_dice'], results['std_dice']))
    results_path = os.path.join(results_folder_path, 'results.pickle')
    with open(results_path, 'wb') as f:
        pickle.dump([results, lung_masks_arr], f)
    return (results, lung_masks_arr)


if __name__ == '__main__':
    predict_all_masks(3)
