import numpy as np
import glob
import sys
import data_inspection
import torch
import matplotlib.pyplot as plt
import re
import os
import pickle
import torch.nn as nn

TRAIN_DATA_DIR_WINDOWS = r"d:\gwaves_data\g2net-gravitational-wave-detection\train"

TRAIN_FILE_NAMES_PICKLE = r"\train_file_names.pkl"
TRAIN_DATA_DIR = None

if sys.platform == "win32":
    TRAIN_DATA_DIR = TRAIN_DATA_DIR_WINDOWS


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=16, stride=4)

    def forward(self, x):
        x = self.conv1(x)
        return x


def get_test_file_names():

    if os.path.isfile(TRAIN_DATA_DIR + TRAIN_FILE_NAMES_PICKLE):

        with open(TRAIN_DATA_DIR + TRAIN_FILE_NAMES_PICKLE, "rb") as f:
            file_name_list = pickle.load(f)
        return file_name_list

    else:

        search_expr = TRAIN_DATA_DIR + r"\**\*.npy"
        print(f"Searching in: {search_expr}")

        file_name_list = glob.glob(search_expr, recursive=True)
        with open(TRAIN_DATA_DIR + TRAIN_FILE_NAMES_PICKLE, "wb") as f:
            pickle.dump(file_name_list, f, protocol=pickle.HIGHEST_PROTOCOL)

        return file_name_list


def create_single_batch(labels_dict, file_names, batch_size=10):

    data_list = []
    labels_list = []
    r = re.compile(r"[0-9a-z]{10}.npy")
    for fn in file_names:
        k = r.search(fn).group(0)
        k = k[0:-4]
        print(f"k: {k[0:-4]}")
        data_list.append(np.load(fn))
        labels_list.append(labels_dict[k])

        if len(data_list) == batch_size:
            break

    batch = np.stack(data_list, axis=0)
    labels = np.array(labels_list)

    return batch, labels


if __name__ == "__main__":

    print("Processing...")

    labels = data_inspection.read_csv(data_inspection.TRAINING_LABELS)
    labels_dict = data_inspection.make_labels_dict(labels[1:])


    test_file_names = get_test_file_names()
    n_test_files = len(test_file_names)
    print(f"Number of files: {n_test_files}")
    print(f"Example file: {test_file_names[0]}")

    example_data_file_name = test_file_names[0]

    batch, labels = create_single_batch(labels_dict, test_file_names, batch_size=10)

    print(f"batch size: {batch.shape} labels: {labels.shape}")
    # example_data_file_name = r"d:\gwaves_data\g2net-gravitational-wave-detection\test\0\0\0\00005bced6.npy"

    batch = torch.from_numpy(batch)
    sn = SimpleNet()
    output = sn.forward(batch.float())

    print(output.shape)

    example_data = np.load(example_data_file_name)
    print(f"Example data shape: {example_data.shape}")


    # plt.plot(example_data[0, :])
    # plt.plot(example_data[1, :])
    # plt.plot(example_data[2, :])
    # plt.ylabel('LIGO H')
    # plt.show()
