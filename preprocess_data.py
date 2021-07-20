from multiprocessing import process
import os
import pickle
import random
import glob
from pathlib import Path
import numpy as np
import multiprocessing

from constants import *
from logger import *
import data_inspection


def dispatch_batched_data_saving(labels_dict,
                                 file_names,
                                 batch_size=200):

    random.shuffle(file_names)

    small_file_names_list = [None for _ in range(batch_size)]
    small_labels_list = [None for _ in range(batch_size)]

    index = 1
    m = 0

    pool = multiprocessing.Pool(processes=4)

    for i in range(len(file_names)):

        fn = file_names[i]
        k = NUMPY_REGEXP.search(fn).group(0)
        k = k[0:-4]

        small_file_names_list[m] = fn
        small_labels_list[m]=labels_dict[k]

        m += 1

        if m == batch_size:

            # p = multiprocessing.Process(target=make_and_save_batched_data, args=(small_labels_list, small_file_names_list, index, BATCHED_DEFAULT_DIR, 0.1))
            # process.append(p)
            # process[-1].start()
            
            pool.apply_async(func=make_and_save_batched_data, args=(small_labels_list, small_file_names_list, index, BATCHED_DEFAULT_DIR, 0.1))
            
            # printd(f"What's inside: {small_labels_list}")

            index += 1
            m = 0

            small_file_names_list = [None for _ in range(batch_size)]
            small_labels_list = [None for _ in range(batch_size)]

    pool.close()
    pool.join()
    

def make_and_save_batched_data(labels_list,
                               file_names,
                               index,
                               batched_data_dir=BATCHED_DEFAULT_DIR,
                               train_test_split=0.1):

    printd(f"Starting for index: {index}")

    n_files = len(file_names)
    data_list = [None for _ in range(n_files)]
    for i in range(len(file_names)):
        # start = time.time()
        fn = file_names[i]
        k = NUMPY_REGEXP.search(fn).group(0)
        k = k[0:-4]

        data_list[i] = np.load(fn)

    Path(batched_data_dir).mkdir(parents=True, exist_ok=True)

    features = np.stack(data_list, axis=0)
    labels = np.array(labels_list).reshape((-1, 1))

    printd(f"At: {i} Saving batch number: {index} features: {features.shape} labels: {labels.shape}")

    # Should switch to torch.save
    np.save(batched_data_dir + r"/" + f"{k}_{index}_features.npy", features)
    np.save(batched_data_dir + r"/" + f"{k}_{index}_labels.npy", labels)

    printd(f"Stopping for index: {index}")


def get_train_file_names():

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


if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')

    print("Processing...")

    labels = data_inspection.read_csv(BASE_DATA_DIR + TRAINING_LABELS)
    labels_dict = data_inspection.make_labels_dict(labels[1:])


    train_file_names = get_train_file_names()
    n_test_files = len(train_file_names)
    print(f"Number of files: {n_test_files}")
    print(f"Example file: {train_file_names[0]}")

    example_data_file_name = train_file_names[0]

    dispatch_batched_data_saving(labels_dict, train_file_names)
