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
from constants import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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

        self.prediction_output = nn.Sigmoid()

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

    def predict(self, x):
        return self.prediction_output(self.forward(x))


def combine_and_match_feature_and_label_files(feature_failes, label_files):

    feature_failes.sort()
    label_files.sort()

    # Make sure that labels match features
    data = list(zip(feature_failes, label_files))
    for d in data:
        i = os.path.basename(d[0]).replace("features", "xkdc")
        l = os.path.basename(d[1]).replace("labels", "xkdc")

        assert i == l, "Files do not match!"

    return data


def train_network(net,
                  device,
                  n_epoch,
                  input_training_data,
                  labels_training_data,
                  input_test_data,
                  labels_test_data):

    train_data = combine_and_match_feature_and_label_files(input_training_data, labels_training_data)
    test_data = combine_and_match_feature_and_label_files(input_test_data, labels_test_data)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(n_epoch):

        running_loss = 0.0
        number_of_batches = 10

        batch_points = [random.randint(0, len(train_data)-1)  for _ in range(number_of_batches)]

        for i in batch_points:

            # if i % 100 == 0:
            #     print(f"We are at {epoch} - {i}")

            input_batch = np.load(train_data[i][0], allow_pickle=True)
            labels_batch = np.load(train_data[i][1], allow_pickle=True)

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


        predicted_labels_list = []
        true_labels_list = []
        for i in range(len(test_data)):

            input_batch = np.load(test_data[i][0], allow_pickle=True)
            labels_batch = np.load(test_data[i][1], allow_pickle=True)


            input_batch = torch.from_numpy(input_batch)

            if device is not None:
                input_batch.to(device)
                input_batch = input_batch.type(torch.cuda.FloatTensor)
            else:
                input_batch = input_batch.type(torch.FloatTensor)

            pred = net.predict(input_batch)
            pred = pred.cpu().detach().numpy()
            predicted_labels_list.append(pred)
            true_labels_list.append(labels_batch)

            # print(f"--> labels_batch: {labels_batch.shape} pred: {pred.shape}")

        predicted_labels = np.concatenate(predicted_labels_list)
        true_labels = np.concatenate(true_labels_list)

        auc = roc_auc_score(true_labels, predicted_labels)

        # print(f"{len(predicted_labels_list)} {len(true_labels_list)} predicted_labels: {predicted_labels.shape} true_labels: {true_labels.shape}")

        print(f"epoch: {epoch} running_loss: {running_loss/number_of_batches} auc: {auc}")


if __name__ == "__main__":

    print("Processing...")

    device = torch.device("cuda:0")

    net = SimpleNet()
    net.to(device)
    # output = sn.forward(batch.float())

    input_data = glob.glob(BATCHED_DEFAULT_DIR + r"/*_features.npy")
    labels_data = glob.glob(BATCHED_DEFAULT_DIR + r"/*_labels.npy")

    input_data.sort()
    labels_data.sort()

    # print(input_data[0:3])
    # print(labels_data[0:3])
    # print(f"{type(input_data)} {type(labels_data)}")

    splited_data = train_test_split(input_data, labels_data, test_size=0.1, random_state=42)
    input_train_data, input_test_data, labels_train_data, labels_test_data = splited_data

    print(f"Train size: {len(input_train_data)} Test size: {len(input_test_data)}")

    train_network(net, device, 100, input_train_data, labels_train_data, input_test_data, labels_test_data)

    index = random.randint(0, len(input_data) - 1)
    print(f"File: {input_data[index]}")

    example_data = np.load(input_data[index])

    example_data = torch.from_numpy(example_data)
    example_data.to(device)
    example_data = example_data.type(torch.cuda.FloatTensor)

    print(f"example data shape: {example_data.shape}")

    labels = net.predict(example_data)

    # print(f"labels: {labels}")

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
