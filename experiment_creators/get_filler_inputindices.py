"""Print decoding analysis input sequence."""
import ast
import os
import pickle
import sys
sys.path.append("../")

from directories import base_dir

def indices_to_words(wordslist, experiment_name):
    data_dir = os.path.join(base_dir, 'data', experiment_name)
    with open(os.path.join(data_dir, "test_analyze.p")) as f:
        indices = pickle.load(f)[0][0].squeeze()

    words = [wordslist[i] for i in indices]
    return words

if __name__ == '__main__':
    experiment_name = sys.argv[1]
    wordslist = ast.literal_eval(sys.argv[2])
    words = indices_to_words(wordslist, experiment_name)
    print(words)
