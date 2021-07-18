import re
import sys

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