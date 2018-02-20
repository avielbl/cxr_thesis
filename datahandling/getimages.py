from aid_funcs.misc import csv_to_list_dict
import os
import shutil
import scipy.io as spio
import numpy as np



findings_dict = {'healthy': 'Healthy',
                 'cardio': 'cardiomegaly',
                 'medias': 'Enlarged mediastinum',
                 'r_pe': 'PE right',
                 'l_pe': 'PE left',
                 'r_ptx': 'ptx right',
                 'l_ptx': 'ptx left',
                 'r_opac': 'Opacity right',
                 'l_opac': 'Opacity left',
                 'pacemaker': 'Defibrilator',
                 'sternotomy': 'Sternotomy',
                 'line': 'Center line catheter',
                 'r_nodules': 'nodules right',
                 'l_nodules': 'nodules left',
                 'edema': 'Pulmonary Edema'}

def getimages(findings, proj='PA', getneg=False, data_repo_path=r'C:\projects\CXR_thesis\data_repo\TRAINING'):
    if not isinstance(findings, (list, tuple)):
        findings = [findings]
    if not isinstance(proj, (list, tuple)):
        proj = [proj]
    root = os.path.join(data_repo_path, 'clean')
    labels_csv_path = [file for file in os.listdir(data_repo_path) if 'db_description' in file and file.endswith('csv')]
    db_path = os.path.join(data_repo_path, labels_csv_path[0])

    db_labels = csv_to_list_dict(db_path)
    patients_dir_lst = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]

    out_list = []
    for finding in findings:
        if finding not in findings_dict.keys():
            print("Given finding wasn't found")
            return []
        finding_fullname = findings_dict[finding]
        values = [d[finding_fullname] for d in db_labels if finding_fullname in d]
        if getneg:
            indecis = [i for i, j in enumerate(values) if j != '1']
        else:
            indecis = [i for i, j in enumerate(values) if j == '1']

        for i in indecis:
            image_found_flag = False
            curr_acc_num = db_labels[i]['Accession']
            if curr_acc_num not in patients_dir_lst:
                continue

            curr_path = os.path.join(root, curr_acc_num)
            imgs_lst = [name for name in os.listdir(curr_path) if os.path.isfile(os.path.join(curr_path, name))]
            for p in proj:
                for image in imgs_lst:
                    if p in image:
                        out_list.append(os.path.join(curr_path, image))
                        image_found_flag = True
            if not image_found_flag:
                print('No relevant image found for patient {}'.format(curr_acc_num))

    return(out_list)


def get_flag(x, category):
    if x[findings_dict[category]] == '1':
        flag = 1
    else:
        flag = 0
    return flag


def sort_ptx_data():
    out_path = r'D:\DICOM_data_repo\CXR\New_train_set_Sheba\for_matlab'
    r_ptx = []
    l_ptx = []
    Image = []
    img_counter = 1
    for d in db_labels:
        curr_path = os.path.join(root, d['Accession'])
        im_list = [name for name in os.listdir(curr_path) if os.path.isfile(os.path.join(curr_path, name))]
        r_ptx_flag = get_flag(d, 'r_ptx')
        l_ptx_flag = get_flag(d, 'l_ptx')

        for im in im_list:
            if 'LAT' not in im:
                # shutil.copyfile(os.path.join(curr_path, im), os.path.join(out_path, 'ptx_data', im))
                r_ptx.append(r_ptx_flag)
                l_ptx.append(l_ptx_flag)
                Image.append(im.split('.')[0])
                print('Copied {} images'.format(img_counter))
                img_counter += 1
    out_file_path = os.path.join(out_path, 'ptx_labels.mat')
    out_dict = {'r_ptx': np.array(r_ptx),
                'l_ptx': np.array(l_ptx),
                'Image': np.array(Image, dtype=np.object)}
    spio.savemat(out_file_path, out_dict)

def sort_data_for_matlab():
    out_path = r'D:\DICOM_data_repo\CXR\New_train_set_Sheba\for_matlab'
    Cardio = []
    Meitsar = []
    ok = []
    Taflitleft = []
    Taflitright = []
    TasninLeft = []
    TasninRight = []
    Image = []

    img_counter = 1
    for d in db_labels:
        curr_path = os.path.join(root, d['Accession'])
        im_list = [name for name in os.listdir(curr_path) if os.path.isfile(os.path.join(curr_path, name))]
        cardio_flag = get_flag(d, 'cardio')
        medias_flag = get_flag(d, 'medias')
        healthy_flag = get_flag(d, 'healthy')
        r_pe_flag = get_flag(d, 'r_pe')
        l_pe_flag = get_flag(d, 'l_pe')
        r_opac_flag = get_flag(d, 'r_opac')
        l_opac_flag = get_flag(d, 'l_opac')

        for im in im_list:
            if 'LAT' not in im:
                shutil.copyfile(os.path.join(curr_path, im), os.path.join(out_path, 'data', im))
                Cardio.append(cardio_flag)
                Meitsar.append(medias_flag)
                ok.append(healthy_flag)
                Image.append(im.split('.')[0])
                Taflitleft.append(l_pe_flag)
                Taflitright.append(r_pe_flag)
                TasninLeft.append(l_opac_flag)
                TasninRight.append(r_opac_flag)
                print('Copied {} images'.format(img_counter))
                img_counter += 1
    out_file_path = os.path.join(out_path, 'labels.mat')
    out_dict = {'Cardio': np.array(Cardio),
                'Meitsar': np.array(Meitsar),
                'ok': np.array(ok),
                'Image': np.array(Image, dtype=np.object),
                'Taflitleft': np.array(Taflitleft),
                'Taflitright': np.array(Taflitright),
                'TasninLeft': np.array(TasninLeft),
                'TasninRight': np.array(TasninRight)}
    spio.savemat(out_file_path, out_dict)
