import os
curr_folder = os.path.dirname(os.path.abspath(__file__))
# List of general params to be used for training CXR lungs segmentation model

# data properties and paths:
im_size = 256
pre_process_params_path = os.path.join(curr_folder, 'pre_process_params.pickle')
optimal_thresh_path = os.path.join(curr_folder, 'optimal_thresh.pickle')
# seg_model_path = 'unet14_02_17.hdf5'
seg_model_path = os.path.join(curr_folder, 'lung_seg_model_10_17_16_02_2017.hdf5')
training_data_dicom = "..\\DATA\\new_Training_Data\\dicom"
training_segmentation_maps = "..\\DATA\\new_Training_Data\\lungsSegmentation\\segmentation_maps"
close_size = 30





