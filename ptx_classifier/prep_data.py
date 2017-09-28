from datahandling.getimages import getimages
from shutil import copy2
import os
import numpy as np

copy_root = r'C:\projects\CXR_thesis\data_repo\TRAINING'

pos_cases = getimages(['r_ptx', 'l_ptx'], ['PA', 'AP'])
pos_path = os.path.join(copy_root, 'pos_cases')

for case in pos_cases:
    copy2(case, pos_path)

neg_cases = getimages(['r_ptx', 'l_ptx'], ['PA', 'AP'], getneg=True)
neg_path = os.path.join(copy_root, 'neg_cases')

neg_samples = np.random.choice(len(neg_cases), size=np.uint16(1.1*len(pos_cases)), replace=False)

sampled_neg_cases = [neg_cases[i] for i in neg_samples]
for i, case in enumerate(sampled_neg_cases):
    copy2(case, neg_path)
    print('copied {}/{}'.format(i, len(sampled_neg_cases)))
