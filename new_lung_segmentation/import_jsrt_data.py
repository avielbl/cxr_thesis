import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import cv2


from image import im_rescale, imresize, get_contour_from_mask
from plot import show_image_with_overlay

in_shape = (2048, 2048) # matrix size
in_dtype = np.dtype('>u2') # big-endian unsigned integer (16bit)
out_shape = (1024, 1024)

jsrt_images_path_in = r'E:\DICOM_data_repo\CXR\external_cxr_dbs\JSRT\All247images'
jsrt_masks_path_in = r'E:\DICOM_data_repo\CXR\external_cxr_dbs\JSRT\scratch'
jsrt_images_path_out = r'C:\projects\CXR_thesis\new_lung_segmentation\jsrt_test_set\images'
jsrt_masks_path_out = r'C:\projects\CXR_thesis\new_lung_segmentation\jsrt_test_set\masks'

images_list = os.listdir(jsrt_images_path_in)

for file in images_list:
    case_name = file.split('.')[0]
    image_path = os.path.join(jsrt_images_path_in, file)
    with open(image_path, 'rb') as f:
        data = np.fromfile(f, in_dtype)
        image = data.reshape(in_shape)
        max_i = np.max(image)
        image = (2 ** np.ceil(np.log2(max_i - 1))) - 1 - image
        image = im_rescale(image, 0, 255)
        image = imresize(image, out_shape)

        l_mask = np.array(Image.open(os.path.join(jsrt_masks_path_in, 'left_lungs', case_name + '.gif')).convert('L'))
        r_mask = np.array(Image.open(os.path.join(jsrt_masks_path_in, 'right_lungs', case_name + '.gif')).convert('L'))
        mask = l_mask + r_mask
        mask[mask > 0] = 255
        mask = imresize(mask, out_shape)
        contour = get_contour_from_mask(mask)
        show_image_with_overlay(image, contour, alpha=0.1, title_str=case_name)
        plt.pause(0.5)
        cv2.imwrite(os.path.join(jsrt_images_path_out, case_name + '.png'), image)
        cv2.imwrite(os.path.join(jsrt_masks_path_out, case_name + '.png'), mask)

