import os
from aid_funcs.plot import show_image_with_overlay
from image import get_contour_from_mask

from utils import *

test_path = r'C:\projects\CXR_thesis\data_repo\TEST\pos_cases\right'
masks_out = r'C:\projects\CXR_thesis\data_repo\TEST\lung_seg'
results_out = r'C:\projects\CXR_thesis\data_repo\TEST\lung_seg\overlays'

test_files = os.listdir(test_path)
model = get_model()
for file in test_files:
    image_name = file.split('.')[0]
    full_path = os.path.join(test_path, file)
    if os.path.isdir(full_path):
        continue
    image, mask = predict(full_path, model)
    contour_mask = get_contour_from_mask(mask)
    result = show_image_with_overlay(image, contour_mask, plot_output=False)
    out_file_name = image_name + '.png'
    mask[mask > 0] = 255
    cv2.imwrite(os.path.join(results_out, out_file_name), result)
    cv2.imwrite(os.path.join(masks_out, out_file_name), mask)
