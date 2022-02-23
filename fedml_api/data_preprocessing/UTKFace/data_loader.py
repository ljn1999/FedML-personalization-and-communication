import logging
import os

import numpy as np
import matplotlib.pyplot as plt  
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

DEFAULT_BATCH_SIZE = 20
DEFAULT_TRAIN_FILE = 'train'
DEFAULT_TEST_FILE = 'test'

AGE = 0
GENDER = 1
FILENAME = 2

IMAGE_SIZE = 28
SPLIT_AGE = 30

def load_partition_data_utkface(dataset, data_dir, batch_size=DEFAULT_BATCH_SIZE):
    # get train and test groups which contains an array of groups,
    # where each group contains elements of [age, gender, filename]
    train_groups, test_groups = get_groups(data_dir)

    # local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    client_idx = 0
    N = 3

    logging.info("\n############ Splitting Clients (START) ############")
    for group_id in range(2): #num_groups
        num_train_images = len(train_groups[group_id])
        i = int(num_train_images / N) # increment for train
        train_image_start_idx = 0 # train image start id
        
        num_test_images = len(test_groups[group_id])
        j = int(num_test_images / N) # increment for test
        test_image_start_idx = 0 # test image start id

        count = 0
        while count < N:
            logging.info("-------------------------------------------------")
            train_image_end_idx = train_image_start_idx+i # train image end id
            if (train_image_end_idx > num_train_images):
                train_image_end_idx = num_train_images-1 # last iteration
            else:
                train_image_end_idx = train_image_start_idx+i
            train_extract = [train_image_start_idx, train_image_end_idx]
            logging.info("train_image_extract indexes= {0}".format(train_extract))

            # test images
            test_image_end_idx = test_image_start_idx+j # test image end id
            if (test_image_end_idx > num_test_images):
                test_image_end_idx = num_test_images-1 # last iteration
            else:
                test_image_end_idx = test_image_start_idx+j
            test_extract = [test_image_start_idx, test_image_end_idx]
            logging.info("test_image_extract indexes= {0}".format(test_extract))

            train_data_local, test_data_local, class_num = get_dataloader(
                data_dir, batch_size, batch_size,
                train_groups[group_id][train_image_start_idx: train_image_end_idx+1],
                test_groups[group_id][test_image_start_idx: test_image_end_idx+1]
            )

            local_data_num = len(train_data_local) + len(test_data_local)
            data_local_num_dict[client_idx] = local_data_num
            logging.info("group_id = %d, client_idx = %d, local_data_number = %d" % (group_id, client_idx, local_data_num))
            logging.info("batch_num_train_local = %d, batch_num_test_local = %d" % (len(train_data_local), len(test_data_local)))
                
            train_data_local_dict[client_idx] = train_data_local
            test_data_local_dict[client_idx] = test_data_local

            train_image_start_idx = train_image_end_idx + 1 # update to next train start id
            test_image_start_idx = test_image_end_idx + 1 # update to next test start id
            client_idx += 1 # update client idx
            count += 1

    logging.info("sum_train_clients_num = %d" % int(2))
    logging.info("############ Splitting Clients (END) ############\n")

    train_data_global = data.DataLoader(
                data.ConcatDataset(
                    list(dl.dataset for dl in list(train_data_local_dict.values()))
                ),
                batch_size=batch_size, shuffle=True)
    train_data_num = len(train_data_global.dataset)
    
    test_data_global = data.DataLoader(
                data.ConcatDataset(
                    list(dl.dataset for dl in list(test_data_local_dict.values()) if dl is not None)
                ),
                batch_size=batch_size, shuffle=True)
    test_data_num = len(test_data_global.dataset)

    
    logging.info("class_num = %d" % class_num)
    train_client_num = len(train_groups)*N
    
    return train_client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


def get_groups(data_dir):
    # get pure train and test records
    test_records = []
    train_records = []

    train_path = os.path.join(data_dir, DEFAULT_TRAIN_FILE)
    test_path = os.path.join(data_dir, DEFAULT_TEST_FILE)

    for filename in os.listdir(train_path):
        record = filename.split('_')
        train_records.append([int(record[AGE]), int(record[GENDER]), filename])

    for filename in os.listdir(test_path):
        record = filename.split('_')
        test_records.append([int(record[AGE]), int(record[GENDER]), filename])
    
    # assign groups
    train_groups = [[],[]] # 2 groups
    test_groups = [[],[]] # 2 groups

    for record in train_records:
        if record[0] < SPLIT_AGE:
            train_groups[0].append(record) # train group 1
        else:
            train_groups[1].append(record) # train group 2
    
    for record in test_records:
        if record[AGE] < SPLIT_AGE:
            test_groups[0].append(record) # test group 1
        else:
            test_groups[1].append(record) # test group 2
    
    return train_groups, test_groups


def get_dataloader(data_dir, train_bs, test_bs, train_groups_extract, test_groups_extract):
    train_path = os.path.join(data_dir, DEFAULT_TRAIN_FILE)
    test_path = os.path.join(data_dir, DEFAULT_TEST_FILE)

    # resize, make it rey, normalize
    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor()
                               ])

    # train
    train_x = np.empty((0,IMAGE_SIZE,IMAGE_SIZE), dtype=np.float32)
    train_y = np.empty((0))
    for record in train_groups_extract:
        image_data = Image.open(os.path.join(train_path, record[FILENAME]))
        transformed_image_data = transform(image_data) # resize and make it gray
        train_x = np.append(train_x, transformed_image_data, axis=0) # data
        train_y = np.append(train_y, record[GENDER]) # labels
    
    # test
    test_x = np.empty((0,IMAGE_SIZE,IMAGE_SIZE), dtype=np.float32)
    test_y = np.empty((0))
    for record in test_groups_extract:
        image_data = Image.open(os.path.join(test_path, record[FILENAME]))
        transformed_image_data = transform(image_data) # resize and make it gray
        test_x = np.append(test_x, transformed_image_data, axis=0) # data
        test_y = np.append(test_y, record[AGE]) # labels
    
    # dataloader
    train_ds = data.TensorDataset(torch.tensor(train_x), torch.tensor(train_y, dtype=torch.long))
    train_dl = data.DataLoader(dataset=train_ds,
                               batch_size=train_bs,
                               shuffle=True,
                               drop_last=False)

    test_ds = data.TensorDataset(torch.tensor(test_x), torch.tensor(test_y, dtype=torch.long))
    test_dl = data.DataLoader(dataset=test_ds,
                                  batch_size=test_bs,
                                  shuffle=True,
                                  drop_last=False)
    
    class_num = len(np.unique(train_y))
    return train_dl, test_dl, class_num