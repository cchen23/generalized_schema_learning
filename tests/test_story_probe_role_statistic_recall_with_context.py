import numpy as np
import os
import pickle

save_dir = os.path.join('/', 'home', 'cc27', 'Thesis', 'generalized_schema_learning', 'data', 'probe_role_statistic_recall_with_context')
with open(os.path.join(save_dir, 'wordlist.p'), 'rb') as f:
    wordlist = pickle.load(f)

noise_word = 'zzz'
noise_index = wordlist.index(noise_word)
context_a_index = wordlist.index('A')
context_b_index = wordlist.index('B')

query_to_input_indices = {
        'qemcee': [6],
        'qfriend': [4],
        'qsubject': [1, 3],
        'qpoet': [8],
        }


def test_distributions():
    def assert_mostly_greater(A, B, tolerance=0.8):
        '''Assert that at least tolerance of indices are greater in A than B.'''
        assert(len(np.where(A > B)[0]) >= len(A) * tolerance)

    with open(os.path.join(save_dir, 'embedding.p'), 'rb') as f:
        embeddings = pickle.load(f)
    for word in embeddings:
        if 'add_05' in word:
            assert_mostly_greater(word['vector'][::2], word['vector'][1::2])
        elif 'subtract_05' in word:
            assert_mostly_greater(word['vector'][1::2], word['vector'][::2])


def test_train_set():
    with open(os.path.join(save_dir, 'train.p'), 'rb') as f:
        X, y = pickle.load(f)

    for example_num in range(len(y)):
        input_sentence = np.array(np.squeeze(X[example_num]), dtype=int)
        query = wordlist[int(input_sentence[-1])]
        context = wordlist[int(input_sentence[0])]
        answer_word_index = int(y[example_num][0])
        answer = wordlist[answer_word_index]
        for word_to_delete in [noise_index, context_a_index, context_b_index]:
            input_sentence = np.delete(np.squeeze(input_sentence), np.where(input_sentence == word_to_delete))
        if query in ['qemcee', 'qfriend']:
            if context == 'A':
                assert('subtract_05_train' in answer)
            else:
                assert('add_05_train' in answer)
        else:
            if context == 'A':
                assert('add_05_train' in answer)
            else:
                assert('subtract_05_train' in answer)
        assert(int(input_sentence[query_to_input_indices[query][0]]) == answer_word_index)


def test_test_set():
    with open(os.path.join(save_dir, 'test.p'), 'rb') as f:
        X, y = pickle.load(f)

    for example_num in range(len(y)):
        input_sentence = np.array(np.squeeze(X[example_num]), dtype=int)
        query = wordlist[int(input_sentence[-1])]
        context = wordlist[int(input_sentence[0])]
        answer_word_index = int(y[example_num][0])
        answer = wordlist[answer_word_index]
        for word_to_delete in [noise_index, context_a_index, context_b_index]:
            input_sentence = np.delete(np.squeeze(input_sentence), np.where(input_sentence == word_to_delete))
        if query in ['qemcee', 'qfriend']:
            if context == 'A':
                assert('subtract_05_test' in answer)
            else:
                assert('add_05_test' in answer)
        else:
            if context == 'A':
                assert('add_05_test' in answer)
            else:
                assert('subtract_05_test' in answer)
        assert(int(input_sentence[query_to_input_indices[query][0]]) == answer_word_index)


def test_flipped_context_test_set():
    with open(os.path.join(save_dir, 'test_flipped_context_distribution.p'), 'rb') as f:
        X, y = pickle.load(f)

    for example_num in range(len(y)):
        input_sentence = np.array(np.squeeze(X[example_num]), dtype=int)
        query = wordlist[int(input_sentence[-1])]
        context = wordlist[int(input_sentence[0])]
        answer_word_index = int(y[example_num][0])
        answer = wordlist[answer_word_index]
        for word_to_delete in [noise_index, context_a_index, context_b_index]:
            input_sentence = np.delete(np.squeeze(input_sentence), np.where(input_sentence == word_to_delete))
        if query in ['qpoet', 'qsubject']:
            if context == 'A':
                assert('subtract_05_train' in answer)
            else:
                assert('add_05_train' in answer)
        else:
            if context == 'A':
                assert('add_05_train' in answer)
            else:
                assert('subtract_05_train' in answer)
        assert(int(input_sentence[query_to_input_indices[query][0]]) == answer_word_index)


def test_unseen_flipped_test_set():
    with open(os.path.join(save_dir, 'test_unseen_flipped_context_distribution.p'), 'rb') as f:
        X, y = pickle.load(f)

    for example_num in range(len(y)):
        input_sentence = np.array(np.squeeze(X[example_num]), dtype=int)
        query = wordlist[int(input_sentence[-1])]
        context = wordlist[int(input_sentence[0])]
        answer_word_index = int(y[example_num][0]) 
        answer = wordlist[answer_word_index]
        for word_to_delete in [noise_index, context_a_index, context_b_index]:
            input_sentence = np.delete(np.squeeze(input_sentence), np.where(input_sentence == word_to_delete))
        if query in ['qpoet', 'qsubject']:
            if context == 'A':
                assert('subtract_05_test' in answer)
            else:
                assert('add_05_test' in answer)
        else:
            if context == 'A':
                assert('add_05_test' in answer)
            else:
                assert('subtract_05_test' in answer)
        assert(int(input_sentence[query_to_input_indices[query][0]]) == answer_word_index)


def test_unseen_no_addition_test_set():
    with open(os.path.join(save_dir, 'test_unseen_no_addition_distribution.p'), 'rb') as f:
        X, y = pickle.load(f)

    for example_num in range(len(y)):
        input_sentence = np.array(np.squeeze(X[example_num]), dtype=int)
        query = wordlist[int(input_sentence[-1])]
        context = wordlist[int(input_sentence[0])]
        answer_word_index = int(y[example_num][0])
        answer = wordlist[answer_word_index]
        for word_to_delete in [noise_index, context_a_index, context_b_index]:
            input_sentence = np.delete(np.squeeze(input_sentence), np.where(input_sentence == word_to_delete))
        assert('add_05' not in answer)
        assert('subtract_05' not in answer)
        assert(int(input_sentence[query_to_input_indices[query][0]]) == answer_word_index)


def test_ambiguous_queried_role_test_set():
    with open(os.path.join(save_dir, 'test_ambiguous_queried_role.p'), 'rb') as f:
        X, y = pickle.load(f)

    for example_num in range(len(y)):
        input_sentence = np.array(np.squeeze(X[example_num]), dtype=int)
        query = wordlist[int(input_sentence[-1])]
        context = wordlist[int(input_sentence[0])]
        for input_index in query_to_input_indices[query]:
            assert((input_sentence[input_index] == noise_index) or (input_sentence[input_index + 1] == noise_index))


def test_ambiguous_all_test_set():
    with open(os.path.join(save_dir, 'test_ambiguous_all.p'), 'rb') as f:
        X, y = pickle.load(f)

    for example_num in range(len(y)):
        input_sentence = np.array(np.squeeze(X[example_num]), dtype=int)
        unique_input_words = np.unique(input_sentence[:-1])
        assert(len(unique_input_words) == 2)
        assert(noise_index in unique_input_words)
