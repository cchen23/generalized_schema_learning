"""Modules to create embedding for experiment."""
import ast
import os
import pickle
import sys

sys.path.append("../")

from directories import base_dir
from embedding_util import create_word_vector

def create_embedding(experiment_data_path):
    """Creates an embedding.

    Args:
        experiment_data_path: The full path of the experiment's data folder.

    Saves (in the experiment_data_path folder):
        embedding.p: A pickle file containing:
                        embedding: A list containing a random vector representing each word in the experiment's corpus.
                        fillers: The fillers used in the experiment (hard-coded).
    """
    print(experiment_data_path)
    with open(os.path.join(experiment_data_path, "Xy_english.txt"), 'rb') as f:
        f.readline() # input
        f.readline() # output
        wordslist = ast.literal_eval(f.readline().split("wordslist: ")[1])

    embedding = []

    for i in range(len(wordslist)):
        word = wordslist[i]
        word_embedding = {}
        word_embedding['index'] = i
        word_embedding['word'] = word
        word_embedding['vector'] = create_word_vector()
        embedding.append(word_embedding)

    with open(os.path.join(experiment_data_path, "embedding.p"), 'wb') as f:
        pickle.dump(embedding, f)

if __name__ == '__main__':
    experiment_foldername = sys.argv[1]
    experiment_data_path = os.path.join(base_dir, "data", experiment_foldername)
    create_embedding(experiment_data_path)
