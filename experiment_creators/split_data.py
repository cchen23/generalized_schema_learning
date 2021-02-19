"""Script to split data by query.

This script splits a dataset into separate datasets, each containing examples of
a single query.
"""
import numpy as np
import os
import pickle
import sys

sys.path.append("../")
from directories import base_dir


def split_query(experiment_name, query):
    """Save a file with inputs containing a specified query."""
    data_path = os.path.join(base_dir, "data", experiment_name)
    train = pickle.load(open(os.path.join(data_path, "test.p")))
    wordslist = pickle.load(open(os.path.join(data_path, "wordslist.p")))
    trainX = train[0]
    trainy = train[1]
    query_wordindex = wordslist.index(query)
    query_storyindices = np.array([True if query_wordindex in x else False for x in trainX], dtype=bool)
    queryX = trainX[query_storyindices]
    queryy = trainy[query_storyindices]

    with open(os.path.join(data_path, 'test_%s.p' % query), 'wb') as f:
        pickle.dump([queryX, queryy], f)


if __name__ == '__main__':
    experiment_name = sys.argv[1]
    querieslist = ['QPOET', 'QSUBJECT', 'QDESSERT', 'QEMCEE', 'QDRINK', 'QFRIEND']
    for query in querieslist:
        split_query(experiment_name, query)
