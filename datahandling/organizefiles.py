import os
import dicom
import shutil
import csv
import matplotlib.pyplot as plt

def get_proj(ds):
    def proj(field):
        if 'lat' in field.lower():
            return 'LAT'
        elif 'pa' in field.lower():
            return 'PA'
        elif 'ap' in field.lower():
            return 'AP'
        else:
            return 'NA'

    if 'ViewPosition' in ds and ds.ViewPosition:
        out = proj(ds.ViewPosition)
    elif 'AcquisitionDeviceProcessingDescription' in ds and ds.AcquisitionDeviceProcessingDescription:
        out = proj(ds.AcquisitionDeviceProcessingDescription)
    else:
        out = 'NA'
    return out

raw_data_path = r'D:\DICOM_data_repo\CXR\TEST\ptx_old_set\all-dicom'
# raw_data_path = r'D:\DICOM_data_repo\CXR\New_test_set_Sheba\raw_data'
# path = r'D:\DICOM_data_repo\CXR\New_test_set_Sheba\raw_data\PNX200217'
out_path = r'D:\DICOM_data_repo\CXR\TEMP\moved_cases'
# out_path = r'D:\DICOM_data_repo\CXR\New_test_set_Sheba\clean'

patients = []
images = []
with open(os.path.join(out_path, 'data_description.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Acc. num', 'Images'])

    for root, dirs, files in os.walk(raw_data_path):
        if len(files) > 0: #run through folders with files only
            for file in files:
                # if '.dcm' in file.lower(): #run through dicom files only
                    accnum = os.path.splitext(file)[0]
                    # accnum = os.path.split(root)[1]
                    # accnum = accnum.split('_')[0] # get 'pure' acc num
                    # accnum = accnum.split(' (')[0] # get 'pure' acc num
                    ds = dicom.read_file(os.path.join(root, file))
                    if 'ProtocolName' in ds and \
                            ('protocol' in ds.ProtocolName.lower() or 'report' in ds.ProtocolName.lower()):
                        continue

                    proj = get_proj(ds) # extract image projection
                    file_name = accnum + '_' + proj
                    curr_path = os.path.join(out_path, accnum)
                    if not os.path.exists(curr_path):
                        os.makedirs(curr_path)

                    if os.path.isfile(os.path.join(curr_path, file_name + '.dcm')):
                        file_name += '2'
                    images.append(file_name)
                    shutil.copyfile(os.path.join(root, file), os.path.join(curr_path, file_name + '.dcm'))

                    if accnum not in patients:
                        patients.append(accnum)
                    writer.writerow([accnum, file_name])
                    print('completed copying {} images of {} patients'.format(len(images), len(patients)))


def load_images(path_list):
    from CXRLoadNPrep import load_dicom
    out_imgs_lst = [load_dicom(path) for path in path_list]
    return out_imgs_lst


def review_images(path_list):
    imgs_arr = load_images(path_list)
    counter = 0
    for img in imgs_arr:
        plt.imshow(img, cmap='gray')
        plt.title(os.path.split(path_list[counter])[1])
        counter += 1
        plt.draw()
        plt.pause(2)