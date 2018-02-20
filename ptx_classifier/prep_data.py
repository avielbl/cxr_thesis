from datahandling.getimages import getimages
from shutil import copy2
import os
import numpy as np

def prep_set(root):
    right_cases = getimages(['r_ptx'], ['PA', 'AP'], data_repo_path=root)
    left_cases = getimages(['l_ptx'], ['PA', 'AP'], data_repo_path=root)
    pos_path = os.path.join(root, 'pos_cases')
    right_path = os.path.join(pos_path, 'right')
    left_path = os.path.join(pos_path, 'left')
    nb_pos = len(right_cases) + len(left_cases)
    for case in right_cases:
        copy2(case, right_path)
    for case in left_cases:
        copy2(case, left_path)

    neg_cases = getimages(['r_ptx', 'l_ptx'], ['PA', 'AP'], getneg=True, data_repo_path=root)
    neg_path = os.path.join(root, 'neg_cases')

    neg_samples = np.random.choice(len(neg_cases), size=np.uint16(1.1*nb_pos), replace=False)

    sampled_neg_cases = [neg_cases[i] for i in neg_samples]
    for i, case in enumerate(sampled_neg_cases):
        copy2(case, neg_path)
        print('copied {}/{}'.format(i, len(sampled_neg_cases)))

if __name__ == '__main__':
    # prep_set(r'C:\projects\CXR_thesis\data_repo\TRAINING')
    prep_set(r'C:\projects\CXR_thesis\data_repo\TEST')