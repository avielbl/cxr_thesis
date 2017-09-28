
pos_images_path = '..\data_repo\TRAINING\pos_cases\';
lung_masks_pos_mat_path = '..\data_repo\TRAINING\lung_masks_pos_cases_gt.mat';
ptx_masks_mat_path = '..\data_repo\TRAINING\ptx_masks_gt.mat';
lung_masks_png_out = '..\data_repo\TRAINING\lung_seg_gt';
ptx_masks_png_out = '..\data_repo\TRAINING\ptx_masks_gt';

files_list = getFilesList(pos_images_path);
N = numel(files_list);
images_names = cell(N,1);
for i=1:N
    tmpStr = files_list(i).name(1:end-4);
    images_names{i}= tmpStr;
end

lung_masks = load(lung_masks_pos_mat_path); lung_masks = lung_masks.lung_masks;
ptx_masks = load(ptx_masks_mat_path); ptx_masks = ptx_masks.ptx_masks;
for i = 1:N
    save_name = strcat(images_names{i}, '.png');
    lung_mask = lung_masks(i).r_lungs_mask + lung_masks(i).l_lungs_mask;
    ptx_mask = ptx_masks(i).ptx_mask;
    if sum(lung_mask(:)) > 0
        imwrite(lung_mask, fullfile(lung_masks_png_out, save_name));
    end
    if sum(ptx_mask(:)) > 0
        imwrite(ptx_mask, fullfile(ptx_masks_png_out, save_name));
    end
end


%% loading all lung seg of the negative case and prepare for presenting in seg tool

neg_imgs_path = '../data_repo\TRAINING\neg_cases\';
lung_seg_path = '../data_repo\TRAINING\lung_seg';
files_list = getFilesList(neg_imgs_path);
N = numel(files_list);
neg_images_names = cell(N,1);
sz = [1024, 1024];
for i=1:N
    tmpStr = files_list(i).name(1:end-4);
    neg_images_names{i}= tmpStr;
end

neg_lung_seg = struct('l_lungs_mask', {}, 'r_lungs_mask', {});
curr_dir = cd(lung_seg_path);
for i=1:N
    file_name = strcat(neg_images_names{i}, '.png');
    % search for existing png file of lung segmentation for this image
    files = dir(file_name);
    r_lungs_mask = false(sz);
    l_lungs_mask = false(sz);
    if isempty(files) % if no file found, fill with empty mask for this case
        neg_lung_seg(i).r_lungs_mask = r_lungs_mask;
        neg_lung_seg(i).l_lungs_mask = l_lungs_mask;
    else
        lung_seg = imread(files.name);
        lung_seg = imresize(lung_seg, sz);
        %seperate into 2 lungs
        cc_lungs = bwconncomp(lung_seg);
        %if there are no 2 objects, don't handle and init masks to empty
        if cc_lungs.NumObjects ~= 2
            neg_lung_seg(i).r_lungs_mask = r_lungs_mask;
            neg_lung_seg(i).l_lungs_mask = l_lungs_mask;
        else
            centers = regionprops(cc_lungs,'Centroid');
            centerline = sz(2)/2;
            left = [centers(1).Centroid(1) > centerline; centers(2).Centroid(1) > centerline];
            right = [centers(1).Centroid(1) < centerline; centers(2).Centroid(1) < centerline];
            left_lung_ind = find(left);
            right_lung_ind = find(right);
            %if seperation to right and left failed, don't handle and init masks to empty
            if isempty(left_lung_ind) || isempty(right_lung_ind)
                neg_lung_seg(i).r_lungs_mask = r_lungs_mask;
                neg_lung_seg(i).l_lungs_mask = l_lungs_mask;
            else
                
                right_lung_ind = cc_lungs.PixelIdxList(right_lung_ind); right_lung_ind = right_lung_ind{1};
                left_lung_ind = cc_lungs.PixelIdxList(left_lung_ind); left_lung_ind = left_lung_ind{1};
                r_lungs_mask(right_lung_ind) = true;
                l_lungs_mask(left_lung_ind) = true;
                neg_lung_seg(i).r_lungs_mask = r_lungs_mask;
                neg_lung_seg(i).l_lungs_mask = l_lungs_mask;
            end
        end
    end
    fprintf('Finished %d/%d segmentations\n', i, N);
end
cd(curr_dir);
lung_masks = neg_lung_seg;
save(fullfile('..\data_repo\TRAINING', 'lung_masks_neg_cases.mat'), 'lung_masks');


%% Save neg lung masks
lung_masks_neg_mat_path = '..\data_repo\TRAINING\lung_masks_neg_cases_gt.mat';
lung_masks = load(lung_masks_neg_mat_path); lung_masks = lung_masks.lung_masks;
for i = 1:N
    save_name = strcat(neg_images_names{i}, '.png');
    lung_mask = lung_masks(i).r_lungs_mask + lung_masks(i).l_lungs_mask;
    if sum(lung_mask(:)) > 0
        imwrite(lung_mask, fullfile(lung_masks_png_out, save_name));
    end
end


