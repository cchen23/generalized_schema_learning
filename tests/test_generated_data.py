import numpy as np
import os
import pickle
import sys

sys.path.append('../')

from hard_coded_things import base_dir

def test_fixed_filler():
    fixed_filler_dir = os.path.join(base_dir, 'data/fixedfiller_AllQs')
    with open(os.path.join(fixed_filler_dir, 'train.p'), 'rb') as f:
        train_X, train_y = pickle.load(f)
    with open(os.path.join(fixed_filler_dir, 'test.p'), 'rb') as f:
        test_X, test_y = pickle.load(f)
    with open(os.path.join(fixed_filler_dir, 'test_unseen.p'), 'rb') as f:
        unseen_test_X, unseen_test_y = pickle.load(f)

    # All overlapping fillers btwn train and test.
    assert(set(np.unique(train_y)) == set(np.unique(test_y)))
    # No overlapping fillers btwn seen and unseen.
    assert(len(set(np.unique(train_y)).intersection(set(np.unique(unseen_test_y)))) == 0)

    # All examples are unique.
    all_X = np.concatenate([train_X, test_X, unseen_test_X], axis=0)
    assert(len(np.unique(all_X, axis=0)) == len(all_X))

    # All answers in input.
    all_y = np.concatenate([train_y, test_y, unseen_test_y], axis=0)
    for example_num in range(len(all_X)):
        assert(all_y[example_num] in all_X[example_num])


def test_variable_filler():
    filler_dir = os.path.join(base_dir, 'data/variablefiller_AllQs')
    with open(os.path.join(filler_dir, 'train.p'), 'rb') as f:
        train_X, train_y = pickle.load(f)
    with open(os.path.join(filler_dir, 'test.p'), 'rb') as f:
        test_X, test_y = pickle.load(f)
    with open(os.path.join(filler_dir, 'test_unseen.p'), 'rb') as f:
        unseen_test_X, unseen_test_y = pickle.load(f)

    assert(set(np.unique(train_y)) == set(np.unique(test_y)) == set(np.unique(unseen_test_y)))

    # All examples are unique.
    all_X = np.concatenate([train_X, test_X], axis=0)
    assert(len(np.unique(all_X, axis=0)) == len(train_X))

    # All answers in input.
    all_y = np.concatenate([train_y, test_y, unseen_test_y], axis=0)
    for example_num in range(len(all_X)):
        assert(all_y[example_num] in all_X[example_num])

if __name__ == '__main__':
    test_fixed_filler()
    test_variable_filler()
