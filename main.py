import numpy as np
import glob
import sys

TEST_DATA_DIR_WINDOWS = r"d:\gwaves_data\g2net-gravitational-wave-detection\test"

TEST_DATA_DIR = None

if sys.platform == "win32":
    TEST_DATA_DIR = TEST_DATA_DIR_WINDOWS

def get_test_file_names():
    search_expr = TEST_DATA_DIR + r"\**\*.npy"
    print(f"Searching in: {search_expr}")
    return glob.glob(search_expr, recursive=True)

if __name__ == "__main__":

    print("Processing...")

    test_file_names = get_test_file_names()
    n_test_files = len(test_file_names)
    print(f"Number of files: {n_test_files}")
    print(f"Example file: {test_file_names[0]}")

    example_data_file_name = test_file_names[0]

    example_data = np.load(example_data_file_name)
    print(f"Example data shape: {example_data.shape}")

