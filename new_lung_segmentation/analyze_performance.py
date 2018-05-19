import json

import cv2
import numpy as np
import os

gt_masks_path = r'C:\projects\CXR_thesis\new_lung_segmentation\jsrt_test_set\masks'
predicted_masks_path = r'C:\projects\CXR_thesis\new_lung_segmentation\jsrt_test_set\results\predicted_masks'

images_list = os.listdir(gt_masks_path)
nb_images = len(images_list)

sens = np.zeros((nb_images,))
spec = np.zeros((nb_images,))
acc = np.zeros((nb_images,))
omega = np.zeros((nb_images,))
dice = np.zeros((nb_images,))

for i in range(nb_images):
    image_name = images_list[i]
    gt_mask = cv2.imread(os.path.join(gt_masks_path, image_name), cv2.IMREAD_GRAYSCALE)
    predicted_mask = cv2.imread(os.path.join(predicted_masks_path, image_name), cv2.IMREAD_GRAYSCALE)
    tp = np.sum(gt_mask * predicted_mask)
    tn = np.sum((1 - gt_mask) * (1 - predicted_mask))
    fp = np.sum((1 - gt_mask) * predicted_mask)
    fn = np.sum(gt_mask * (1 - predicted_mask))
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
json.dump(results, open(r'C:\projects\CXR_thesis\new_lung_segmentation\jsrt_test_set\results\results.json', 'w'))
