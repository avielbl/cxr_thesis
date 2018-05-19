import os
import time
from aid_funcs.plot import show_image_with_overlay
from image import get_contour_from_mask

from utils import *

test_path = r'C:\projects\CXR_thesis\new_lung_segmentation\jsrt_test_set\images'
masks_out = r'C:\projects\CXR_thesis\new_lung_segmentation\jsrt_test_set\results\predicted_masks'
results_out = r'C:\projects\CXR_thesis\new_lung_segmentation\jsrt_test_set\results'

test_files = os.listdir(test_path)
model = get_model()
for file in test_files:
    full_path = os.path.join(test_path, file)
    if os.path.isdir(full_path):
        continue
    image, mask = predict(full_path, model)
    mask[mask>0] = 255
    contour_mask = get_contour_from_mask(mask)
    result = show_image_with_overlay(image, contour_mask, plot_output=False)
    cv2.imwrite(os.path.join(results_out, file), result)
    cv2.imwrite(os.path.join(masks_out, file), mask)

