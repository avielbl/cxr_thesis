import os
import numpy as np
np.random.seed(1)
from aid_funcs.misc import zip_save, roc_plotter
from predict_ptx import batch_predict
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
test_path = r'C:\projects\CXR_thesis\data_repo\TEST'


def compute_auc(predictions, right_labels, left_labels):
    '''
    todo:
    plot, show and save roc and results for right, left and total
    '''
    pass


def analyze_results():
    right_res, left_res, neg_res = zip_load('test_res.h5')
    right_labels, left_labels, neg_labels = zip_load('test_labels.h5')
    nb_samples = len(right_labels) + len(left_labels) + len(neg_labels)
    scores = np.zeros((nb_samples, 2))
    combined_res = right_res + left_res + neg_res
    combined_labels = np.concatenate((right_labels, left_labels, neg_labels))
    for i, res in enumerate(combined_res):
        scores[i, 0] = res[1].l_coverage
        scores[i, 1] = res[1].r_coverage

    left_auc = roc_plotter(combined_labels[:, 0], scores[:, 0], 'Left ROC', True)
    right_auc = roc_plotter(combined_labels[:, 1], scores[:, 1], 'Right ROC', True)
    total_auc = roc_plotter(np.concatenate((combined_labels[:, 0], combined_labels[:, 1])),
                            np.concatenate((scores[:, 0], scores[:, 1])), 'Total ROC', True)
    per_patient_auc = roc_plotter(combined_labels[:, 0] + combined_labels[:, 1],
                                  np.max(scores, ), 'Per-patient ROC', True)
    return



def gen_labels(path, label=None):
    imgs_lst = os.listdir(path)
    nb_imgs = len(imgs_lst)
    labels = np.zeros((nb_imgs, 2), dtype=np.uint8)
    if label is None:
        return labels
    if label == 'left':
        labels[:, 0] = 1
    elif label == 'right':
        labels[:, 1] = 1
    return labels


def main():
    right_path = os.path.join(test_path, 'pos_cases', 'right')
    right_labels = gen_labels(right_path, 'right')
    left_path = os.path.join(test_path, 'pos_cases', 'left')
    left_labels = gen_labels(right_path, 'left')
    neg_path = os.path.join(test_path, 'neg_cases')
    neg_labels = gen_labels(neg_path)

    right_res = batch_predict(right_path, right_labels)
    left_res = batch_predict(left_path, left_labels)
    neg_res = batch_predict(neg_path, neg_labels)
    zip_save((right_res, left_res, neg_res), 'test_res.h5')
    zip_save((right_labels, left_labels, neg_labels), 'test_labels.h5')

if __name__ == '__main__':
    main()
    analyze_results()