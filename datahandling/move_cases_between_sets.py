import os
import shutil

src_path = r'D:\DICOM_data_repo\CXR\TRAINING\clean'
dst_path = r'D:\DICOM_data_repo\CXR\TEMP\moved_cases'
lst_of_cases_path = r'D:\DICOM_data_repo\CXR\TEMP\cases_to_copy_from_train_to_test.txt'

with open(lst_of_cases_path) as f:
    counter = 1
    cases = [line.rstrip('\n') for line in f]
    for case in cases:
        curr_path = os.path.join(src_path, case)
        if os.path.isdir(curr_path):
            shutil.move(curr_path, dst_path)
            print('({}): moved: {}'.format(counter, case))
        else:
            print('******** Couldn\'t find:: {}'.format(case))
        counter += 1


train_cases_in_db_path = r"D:\DICOM_data_repo\CXR\TEMP\train_cases_in_csv.txt"
train_path = r'D:\DICOM_data_repo\CXR\training\clean'
missing_cases_in_dir = []
missing_cases_in_db = []
with open(train_cases_in_db_path) as f:
    cases = [line.rstrip('\n') for line in f]
    cases = [x.lower() for x in cases]
    cases_in_dir = os.listdir(train_path)
    cases_in_dir = [x.lower() for x in cases_in_dir]
    print('Cases in directory without labeling:')
    for dir_case in cases_in_dir:
        if dir_case not in cases:
            missing_cases_in_db.append(dir_case)
            print(dir_case)

    print('Cases with labeling without dicom:')
    for db_case in cases:
        if db_case not in cases_in_dir:
            missing_cases_in_dir.append(db_case)
            print(db_case)
