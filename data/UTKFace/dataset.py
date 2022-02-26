import shutil
import os
import random

'''
Before running this file:
1. Create a new folder called UTKFace under "project_directory/data".
2. Download the UTKFace dataset and unzip. You should aim for this directory structure: "project_directory/data/UTKFace/UTKFace"
3. In the terminal, cd to "project_directory/data/UTKFace".
4. Now run this file! If successful, you should see the dataset split into "train" and "test" folders.
'''

path = os.path.join(os.getcwd(), "UTKFace")
data_by_age = {}

# scan data by age
for filename in os.listdir(path):
    age_end_idx = filename.find('_')
    age = filename[:age_end_idx]
    if age not in data_by_age:
        data_by_age[age] = []
        data_by_age[age].append(filename)
    else:
        data_by_age[age].append(filename)

# shuffle the images within the same age
for i in data_by_age:
    random.shuffle(data_by_age[i])

# create train and test dataset
os.makedirs(path+'/train')
os.makedirs(path+'/test')

# split the dataset into 70% for training and 30% for test in each age
for age in data_by_age:
    len_total_data_in_age = len(data_by_age[age])
    len_train_data_in_age = int(len(data_by_age[age])*0.7)

    count = 0
    for f in data_by_age[age]:
        if count < len_train_data_in_age:
            shutil.move(os.path.join(path, f), os.path.join(path, "train", f))
        else:
            shutil.move(os.path.join(path, f), os.path.join(path, "test", f))
        count+=1