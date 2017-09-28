%% training set
seg_path = 'C:\Projects\Algorithm_Dev\CXR\DATA\new_Training_Data\ptx_classification\ptxMasks_training_gt.mat';
ptx_masks = load(seg_path); ptx_masks = ptx_masks.ptx_masks;
N = numel(ptx_masks);

im_path = 'C:\Projects\Algorithm_Dev\CXR\DATA\new_Training_Data\dicom';
dir_list = getFilesList(im_path);

out_path = 'C:\Projects\Algorithm_Dev\CXR\DATA\new_Training_Data\ptx_classification\ptx_maps\';
for i = 1:N
    file_path = [out_path, dir_list(i).name(1:end-4), '.png'];
    mask = ptx_masks(i).ptx_mask;
    imwrite(mask, file_path);
end

%% testing set
seg_path = 'C:\Projects\Algorithm_Dev\CXR\DATA\Testing_Data\ptx_masks_gt.mat';
ptx_masks = load(seg_path); ptx_masks = ptx_masks.ptx_masks;
N = numel(ptx_masks);

im_path = 'C:\Projects\Algorithm_Dev\CXR\DATA\Testing_Data\all-dicom';
dir_list = getFilesList(im_path);

out_path = 'C:\Projects\Algorithm_Dev\CXR\DATA\Testing_Data\ptx_maps\';
for i = 1:N
    file_path = [out_path, dir_list(i).name(1:end-4), '.png'];
    mask = ptx_masks(i).ptx_mask;
    imwrite(mask, file_path);
end