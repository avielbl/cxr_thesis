import numpy as np
import os
from aid_funcs.misc import zip_load

training_path = r'C:\projects\CXR_thesis\data_repo\TRAINING'
model_path = r'ptx_model_13_38_30_09_2017.hdf5'

im_size = 1024
patch_sz = 32
smooth = 1.
max_num_of_patches = 4000000


def is_ptx_case(ptx_mask):
    if ptx_mask is None or np.sum(ptx_mask) == 0:
        return False
    else:
        return True


def train_val_partition():
    data_lst = zip_load(os.path.join(training_path, 'train_set.pkl'))
    nb_train_total = len(data_lst)
    val_idx = np.random.choice(range(nb_train_total), int(0.3 * nb_train_total))

    # Partition to train and val sets
    n_val = len(val_idx)
    n_train = nb_train_total - n_val
    print('Partition to validation (n={}) and training (n={}) sets'.format(n_val, n_train))
    val_data_lst = []
    train_data_lst = []
    for i in range(nb_train_total):
        if i in val_idx:
            val_data_lst.append(data_lst[i])
        else:
            # if i not in val_idx:
            train_data_lst.append(data_lst[i])
    return train_data_lst, val_data_lst