import numpy as np
import glob
import sys
import data_inspection
import torch
import matplotlib.pyplot as plt
import re
import os
import random
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import time

TRAIN_DATA_DIR_WINDOWS = r"d:\gwaves_data\g2net-gravitational-wave-detection\train"
TRAIN_DATA_DIR_LINUX = r"/media/dawid/My_Passport/gwaves_data/g2net-gravitational-wave-detection/train"

BASE_DATA_DIR_LINUX = r"/media/dawid/My_Passport/gwaves_data/g2net-gravitational-wave-detection"
BATCHED_DEFAULT_DIR = r"/home/dawid/Coding/gwaves_batched_data"

TRAIN_FILE_NAMES_PICKLE_WINDOWS = r"\train_file_names.pkl"
TRAIN_FILE_NAMES_PICKLE_LINUX = r"/train_file_names.pkl"
TRAINING_LABELS = r"/training_labels.csv"
NUMPY_SEARCH_PATTERN_LINUX = r"/**/*.npy"
NUMPY_REGEXP = re.compile(r"[0-9a-z]{10}.npy")
TRAIN_DATA_DIR = None
BASE_DATA_DIR = None
TRAIN_FILE_NAMES_PICKLE = None
NUMPY_SEARCH_PATTERN = None

if sys.platform == "win32":
    TRAIN_DATA_DIR = TRAIN_DATA_DIR_WINDOWS
    TRAIN_FILE_NAMES_PICKLE = TRAIN_FILE_NAMES_PICKLE_WINDOWS
elif sys.platform == "linux":
    TRAIN_DATA_DIR = TRAIN_DATA_DIR_LINUX
    TRAIN_FILE_NAMES_PICKLE = TRAIN_FILE_NAMES_PICKLE_LINUX
    BASE_DATA_DIR = BASE_DATA_DIR_LINUX
    NUMPY_SEARCH_PATTERN = NUMPY_SEARCH_PATTERN_LINUX

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=16, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorm1 = nn.BatchNorm1d(num_features=8)
        
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=8, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorm2 = nn.BatchNorm1d(num_features=8)

        self.output = nn.Linear(in_features=1008, out_features=1)

    def forward(self, x):

        # Layer one
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.batchnorm1(x)

        # Layer two
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.batchnorm2(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x

def train_network(net, device=None, n_epoch=10):

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    input_data = glob.glob(BATCHED_DEFAULT_DIR + r"/*_features.npy")
    labels_data = glob.glob(BATCHED_DEFAULT_DIR + r"/*_labels.npy")
    print(input_data)
    print(labels_data)

    input_data.sort()
    labels_data.sort()

    train_data = list(zip(input_data, labels_data))
    for td in train_data:
        i = os.path.basename(td[0]).replace("features", "xkdc")
        l = os.path.basename(td[1]).replace("labels", "xkdc")

        assert i == l, "Files do not match!"

    for epoch in range(n_epoch):

        running_loss = 0.0
        for i in range(len(train_data)):

            input_batch = np.load(train_data[i][0])
            labels_batch = np.load(train_data[i][1])

            input_batch = torch.from_numpy(input_batch)
            labels_batch = torch.from_numpy(labels_batch)

            if device is not None:
                input_batch.to(device)
                labels_batch.to(device)
                input_batch = input_batch.type(torch.cuda.FloatTensor)
                labels_batch = labels_batch.type(torch.cuda.FloatTensor)
            else:
                input_batch = input_batch.type(torch.FloatTensor)
                labels_batch = labels_batch.type(torch.FloatTensor)

            optimizer.zero_grad()

            outputs = net(input_batch)
            # print(outputs)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            # running_loss += loss.item()
            # if i % 20 == 0:   
            #     print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 20))
            #     running_loss = 0.0
            running_loss += loss.item() 
        print(f"epoch: {epoch} running_loss: {running_loss/len(train_data)}")



def get_test_file_names():

    if os.path.isfile(TRAIN_DATA_DIR + TRAIN_FILE_NAMES_PICKLE):

        with open(TRAIN_DATA_DIR + TRAIN_FILE_NAMES_PICKLE, "rb") as f:
            file_name_list = pickle.load(f)
        return file_name_list

    else:

        search_expr = TRAIN_DATA_DIR + NUMPY_SEARCH_PATTERN
        print(f"Searching in: {search_expr}")

        file_name_list = glob.glob(search_expr, recursive=True)
        with open(TRAIN_DATA_DIR + TRAIN_FILE_NAMES_PICKLE, "wb") as f:
            pickle.dump(file_name_list, f, protocol=pickle.HIGHEST_PROTOCOL)

        return file_name_list


def create_single_batch(labels_dict, file_names, batch_size=10):

    data_list = []
    labels_list = []
    for fn in file_names:
        k = NUMPY_REGEXP.search(fn).group(0)
        k = k[0:-4]
        print(f"k: {k[0:-4]}")
        data_list.append(np.load(fn))
        labels_list.append(labels_dict[k])

        if len(data_list) == batch_size:
            break

    batch = np.stack(data_list, axis=0)
    labels = np.array(labels_list)

    return batch, labels


def make_and_save_batched_data(labels_dict,
                               file_names,
                               batch_size=100,
                               batched_data_dir=BATCHED_DEFAULT_DIR,
                               train_test_split=0.1):

    random.shuffle(file_names)

    data_list = [None for _ in range(batch_size)]
    labels_list = [None for _ in range(batch_size)]
    m = 0
    index = 1
    for i in range(len(file_names)):
        # start = time.time()
        fn = file_names[i]
        k = NUMPY_REGEXP.search(fn).group(0)
        k = k[0:-4]

        data_list[m] = np.load(fn)
        labels_list[m]=labels_dict[k]

        m += 1

        if m == batch_size:
            Path(batched_data_dir).mkdir(parents=True, exist_ok=True)

            features = np.stack(data_list, axis=0)
            labels = np.array(labels_list).reshape((-1, 1))

            print(f"At: {i} Saving batch number: {index} features: {features.shape} labels: {labels.shape}")

            # Should switch to torch.save
            np.save(batched_data_dir + r"/" + f"{k}_{index}_features.npy", features)
            np.save(batched_data_dir + r"/" + f"{k}_{index}_labels.npy", labels)

            data_list = [None for _ in range(batch_size)]
            labels_list = [None for _ in range(batch_size)]

            index += 1
            m = 0

        # stop = time.time()
        # print(f"Elapsed time: {stop - start}")

    # For the time being we ignore len(data_list) != 0



if __name__ == "__main__":

    print("Processing...")

    labels = data_inspection.read_csv(BASE_DATA_DIR + TRAINING_LABELS)
    labels_dict = data_inspection.make_labels_dict(labels[1:])


    test_file_names = get_test_file_names()
    n_test_files = len(test_file_names)
    print(f"Number of files: {n_test_files}")
    print(f"Example file: {test_file_names[0]}")

    example_data_file_name = test_file_names[0]

    batch, labels = create_single_batch(labels_dict, test_file_names, batch_size=10)

    print(f"batch size: {batch.shape} labels: {labels.shape}")
    # example_data_file_name = r"d:\gwaves_data\g2net-gravitational-wave-detection\test\0\0\0\00005bced6.npy"

    # make_and_save_batched_data(labels_dict, test_file_names)

    # batch = torch.from_numpy(batch)
    sn = SimpleNet()
    # output = sn.forward(batch.float())

    train_network(sn, device=None, n_epoch=100)

    # loss = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(sn.parameters(), lr=0.001)

    # print(output.shape)

    # example_data = np.load(example_data_file_name)
    # print(f"Example data shape: {example_data.shape}")


    # plt.plot(example_data[0, :])
    # plt.plot(example_data[1, :])
    # plt.plot(example_data[2, :])
    # plt.ylabel('LIGO H')
    # plt.show()
