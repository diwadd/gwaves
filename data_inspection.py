import csv
import numpy as np
import pandas as pd

TRAINING_LABELS = r"d:\gwaves_data\g2net-gravitational-wave-detection\training_labels.csv"

def read_csv(filename):

    with open(filename, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data


if __name__ == "__main__":

    id_target_df = pd.read_csv(TRAINING_LABELS)
    # id_target_df = pd.DataFrame(id_target)

    # id_target = np.array(id_target[1:])

    # target = id_target[:, 1]


    print(id_target_df.head())
    print(id_target_df.columns)
    print(id_target_df["target"].value_counts())