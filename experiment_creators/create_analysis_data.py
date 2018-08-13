"""Create data for analysis."""
import os
import numpy as np
import pickle
import sys

sys.path.append("../")
from directories import base_dir

def create_analysis_data(experiment_name):
    data_path = os.path.join(base_dir, "data", experiment_name)
    with open(os.path.join(data_path, "test.p"), "rb") as f:
      data = pickle.load(f)

    num_repeats = 100
    X = data[0]
    padding_index = X[0][-3]
    padding_counts = [np.count_nonzero(X[i] == padding_index) for i in range(X.shape[0])]
    selected_index = np.argmin(padding_counts)
    X_selected = np.expand_dims(X[selected_index], axis=0)
    X_selected = np.repeat(X_selected, num_repeats, axis=0)
    y_selected = np.expand_dims(data[1][selected_index], axis=0)
    y_selected = np.repeat(y_selected, num_repeats, axis=0)

    with open(os.path.join(data_path, "test_analyze.p"), "wb") as f:
        pickle.dump((X_selected, y_selected), f)

if __name__ == '__main__':
    experiment_name = sys.argv[1]
    create_analysis_data(experiment_name)
