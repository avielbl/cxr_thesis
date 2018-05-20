import os
import shutil

import numpy as np

from image import get_contour_from_mask
from plot import show_image_with_overlay
import matplotlib.pyplot as plt
from utils import *

plt.ion()
orig_images_path = r'C:\projects\CXR_thesis_backup\data_repo\NIH\images'
out_images_path = r'C:\projects\CXR_thesis\new_lung_segmentation\train_set\images'
out_masks_path = r'C:\projects\CXR_thesis\new_lung_segmentation\train_set\masks'

for_manual_segm_path = r'C:\projects\CXR_thesis\new_lung_segmentation\manual'

images_list = os.listdir(orig_images_path)
nb_images = len(images_list)
model = get_model()

for i in np.arange(nb_images-1, 0, -1):
    image_path = os.path.join(orig_images_path, images_list[i])
    image, mask = predict(image_path, model)
    contour = get_contour_from_mask(mask)
    show_image_with_overlay(image, contour, title_str=images_list[i])
    plt.pause(0.5)
    flag = True
    while flag:
        response = input('good result (y) or bad (n)?')
        if response.lower() == 'y':
            shutil.copy2(image_path, out_images_path)
            mask[mask > 0] = 255
            cv2.imwrite(os.path.join(out_masks_path, images_list[i]), mask)
            flag = False
        elif response.lower() == 'n':
            shutil.copy2(image_path, for_manual_segm_path)
            flag = False
        else:
            print("Response only 'y' or 'n'")
