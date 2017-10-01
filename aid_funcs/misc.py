import gzip
import pickle
import h5py
import numpy as np
import io
import csv
import os
import dicom
import scipy.io as spio


def zip_save(object, filename, protocol = -1):
    """Saves a compressed object to disk
    """
    file = gzip.GzipFile(filename, 'wb')
    pickle.dump(object, file, protocol)
    file.close()


def zip_load(filename):
    """Loads a compressed object from disk
    """
    file = gzip.GzipFile(filename, 'rb')
    object = pickle.load(file)
    file.close()
    return object


def load_mat_h5py(path):
    """
    Reads large *.mat files which were saved using ver 7.3 (HDF5 compressed)

    :param path: path to the mat file as string
    :return: dictionary with imported data
    """
    arrays = {}
    with h5py.File(path) as f:
        for k, v in f.items():
            arrays[k] = np.array(v)
    if len(arrays) == 1:
        return list(arrays.values())[0]
    else:
        return arrays


def loadmat(path, var_name=None, ret_names=False):
    """
    Perform 'smart' loading of mat files which returns the variables saved in Matlab without all the 'decorations'
    If only a single variable was stored in the file, it will be returend as was saved.
    If several variables were saved, they will be returned in a list.
    It is possible to receive a corresponding list of variables names.

    :param path: path string to the mat file
    :param var_name: (optional) string of a specific variable name to extract
    :param ret_names: (optional) flag indicating if to return also a list of variable names in the mat file
    :return: (var_names), var_values
    """

    raw_dict = spio.loadmat(path)
    if var_name is None:
        values = []
        names = []
        for val in raw_dict.items():
            if val[0] == '__globals__' or val[0] == '__header__' or val[0] == '__version__':
                continue
            values.append(val[1])
            names.append(val[0])
        if len(values) == 1:
            values = values[0]
            names = names[0]
    else:
        values = raw_dict[var_name]
        names = var_name

    if ret_names:
        return names, values
    else:
        return values


def right_broadcast(arr, target):
    """
    add singleton dimensions to the right of given 'arr' based on dimensions of given 'target'.
    This allows broadcasting of 'arr' so it could be manipulated with target

    Example:
        arr.shape --> (10,)
        target.shape --> (10,1,256,256)
        return_arr.shape --> (10,1,1,1)
    """
    return arr.reshape(arr.shape + (1,) * (target.ndim - arr.ndim))


def csv_to_list_dict(path):
    """
    Reads a *.csv file and generate a list of dictionaries based on the header row fields

    :param path: path to the csv as string
    :return: list of dictionaries
    """
    with open(path, 'r', newline='', encoding='utf-8') as f:
        # manually removing 'empty space' character joined to header row first element as '\ufeff'
        header = csv.reader(f)
        categories = next(header)
        categories[0] = categories[0].split('\ufeff')[1]

    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, fieldnames=categories)
        my_list = list(reader)
    my_list.pop(0)
    return my_list


def loadDCM(dataFolderName):
    """

    :param dataFolderName: 'data' folder of the dcm files
    The function look for *.dcm files in the folder, if there aren't any, the function take all the files in the
      folder (that aren't folders)
    Then the function sort the slices b
    :return: vol - 3d volume of the CT with HU values the dimensions are (Z,rows,columns) to change to (rows,columns,Z)
    use transpose: vol = loadDCM(folderName).transpose(1,2,0)
    """
    files = [f for f in os.listdir(dataFolderName) if f.endswith('.dcm')]
    if len(files) == 0:
        files = [f for f in os.listdir(dataFolderName) if not os.path.isdir(os.path.join(dataFolderName, f))]
    allDcms = list()
    zpos = list()
    for i in np.arange(len(files)):
        allDcms.append(dicom.read_file(os.path.join(dataFolderName, files[i])))
        zpos.append(allDcms[-1].ImagePositionPatient[2])
    sortedIdx = np.argsort(-np.array(zpos))
    vol = np.zeros((len(sortedIdx),allDcms[0].Rows,allDcms[0].Columns))
    for idx in sortedIdx:
        vol[idx] = allDcms[idx].pixel_array
    vol = vol*allDcms[0].RescaleSlope+allDcms[0].RescaleIntercept
    return vol


def mm_to_pixels(pixel_size_in_mm, mm_size):
    """ Generating num of pixels for a given physical size based on pixel size in mm (slice spacing)"""

    out = np.round(mm_size / pixel_size_in_mm).astype(np.uint16)
    return out


def normalize_slices(img_arr):
    """
    Utility function for normalizing all slices in a volume so each will have mean=0 and std=1
    It is roubost to any array of shape larger than 2-d assuming that last 2 axis represents the image
    """

    img_arr = img_arr.astype('float32')
    nb_dim = img_arr.ndim # being robust to any array shape of larger than 2-d
    mean_val = np.mean(img_arr, axis=(nb_dim-2, nb_dim-1))
    mean_val = right_broadcast(mean_val, img_arr)
    std_val = np.std(img_arr, axis=(nb_dim-2, nb_dim-1))
    std_val = right_broadcast(std_val, img_arr)
    img_arr -= mean_val
    img_arr /= std_val
    return img_arr

def save_to_h5(data, file_name):
    with h5py.File(file_name, 'w') as hf:
        hf.create_dataset('name-of-dataset', data=data)

def load_from_h5(file_name):
    with h5py.File(file_name, 'r') as hf:
        data = hf['name-of-dataset'][:]

    return data
