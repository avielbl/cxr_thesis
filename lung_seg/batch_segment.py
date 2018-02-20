import sys
import numpy as np
import matplotlib.pyplot as plt
import CXRLoadNPrep as clp
import os
import time
import pickle
from drawnow import drawnow
import aid_funcs.image as image
from aid_funcs.plot import show_image_with_overlay
from aid_funcs.keraswrapper import load_model
import keras.models
from scipy import ndimage
from skimage import measure, morphology
from scipy.misc import imsave
from scipy.io import savemat
from lung_seg.predict import predict as lung_seg_predict
from lung_seg.utilfuncs import load_image, im_size

# from lateral_frontal_detection import predict as lat_predict
# from lateral_frontal_detection import model_path as lat_model_path

# lat_model = load_model(lat_model_path)

def batch_segment(in_path, out_path, save_mat_flag=False):
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(curr_folder, 'lung_seg_model_10_17_16_02_2017.hdf5')
    import keras.backend as K
    K.set_image_data_format('channels_first')
    model = load_model(model_path, custom_objects='dice_coef_loss')
    images_dir = os.listdir(in_path)
    n = sum([len(files) for r, d, files in os.walk(in_path)])
    all_seg = {}
    if not os.path.exists(os.path.join(out_path, 'overlays')):
        os.makedirs(os.path.join(out_path, 'overlays'))
    if save_mat_flag:
        if not os.path.exists(os.path.join(out_path, 'mat')):
            os.makedirs(os.path.join(out_path, 'mat'))
    i = 1
    for root, dirs, files in os.walk(in_path):
        if len(files) > 0:  # run through folders with files only
            for file in files:
                start_time = time.time()
                im_name = os.path.splitext(file)[0]
                img = load_image(os.path.join(root, file))
                if not isinstance(img, (np.ndarray, np.generic) ):
                    print('skipping {} as it isn\'t a DICOM file'.format(im_name))
                    continue
                # if 'lat' in lat_predict(os.path.join(root, file), lat_model):
                #     continue
                # Actual prediction
                res = lung_seg_predict(os.path.join(root, file), model)
                # if np.sum(res['r_lung_mask']) == 0 or np.sum(res['l_lung_mask']) == 0:
                #     print('segmentation failed in image {} number {:d}'.format(i, im_name))
                all_seg[im_name] = res
                lungs_both = res.r_lung_mask + res.l_lung_mask

                # Presenting and saving overlay image
                show_image_with_overlay(img=np.squeeze(img), overlay=256 * lungs_both, title_str=im_name)
                overlay_path = os.path.join(out_path, 'overlays', im_name) + '.png'
                plt.savefig(overlay_path, bbox_inches='tight')

                # saving seg as mat file if required
                if save_mat_flag:
                    mat_path = os.path.join(out_path, 'mat', im_name) + '.mat'
                    savemat(mat_path, res)

                # saving segmentation map as png
                im_name = os.path.join(out_path,im_name) + '.png'
                imsave(im_name, lungs_both)

                print('Completed segmentation %i/%i in %.2f seconds' % (i+1, n, time.time() - start_time))
                i += 1

    if save_mat_flag:
        savemat(os.path.join(out_path, 'all_lung_seg.mat'), all_seg)


# def predict_cxr_lung_segm(img):
#     """
#
#     :param im_path:
#     :return:
#     """
#     # try:
#     scores = model.predict(img, verbose=0)
#     scores = np.squeeze(scores)
#     predicted_mask = np.zeros_like(scores)
#     predicted_mask[scores >= lung_seg_score_thresh] = 1
#     label_im, nb_labels = ndimage.label(predicted_mask)
#     # remove small objects
#     if nb_labels > 2:
#         areas = ndimage.sum(predicted_mask, label_im, range(nb_labels + 1))
#         sorted_areas = np.sort(areas)
#         smallest_lung_ares = sorted_areas[-2]
#         mask_size = areas < smallest_lung_ares
#         remove_pixel = mask_size[label_im]
#         label_im[remove_pixel] = 0
#     # closing gaps along lungs contour and fill holes
#     r_lung_mask = np.zeros_like(label_im)
#     l_lung_mask = np.zeros_like(label_im)
#     labels = np.unique(label_im)
#     labels = labels[1:]
#     for lung in labels:
#         close_size = 30
#         curr_lung = np.zeros_like(label_im)
#         curr_lung[label_im == lung] = 1
#         se = morphology.disk(close_size)
#         pad_width = ((close_size, close_size), (close_size, close_size))
#         padded_mask = np.pad(curr_lung, pad_width, mode='constant')
#         curr_lung = morphology.binary_closing(padded_mask, se)
#         curr_lung = curr_lung[close_size:-close_size, close_size:-close_size]
#         curr_lung = morphology.remove_small_holes(curr_lung, im_size ** 2 / 3)
#         m = measure.moments(np.uint8(curr_lung))
#         cc = m[1, 0] / m[0, 0]
#         if cc < im_size/2:
#             r_lung_mask[curr_lung] = 1
#         else:
#             l_lung_mask[curr_lung] = 1
#     return {'r_lung_mask': r_lung_mask, 'l_lung_mask': l_lung_mask}
#     # except:
#     #     e = sys.exc_info()[0]
#     #     return 1


if __name__ == '__main__':
    in_path = 'D:\Data from Sheba\\320oldDicoms\\all_dataset'
    out_path = 'D:\Data from Sheba\\320oldDicoms\lungs_ segmentations'
    batch_segment(in_path, out_path)
