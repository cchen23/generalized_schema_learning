import numpy as np
import os
import pickle

save_dir = os.path.join('/', 'home', 'cc27', 'Thesis', 'generalized_schema_learning', 'data', 'probe_role_statistic_recall_normalize_75')
with open(os.path.join(save_dir, 'wordlist.p'), 'rb') as f:
    wordlist = pickle.load(f)

noise_word = 'zzz'
noise_index = wordlist.index(noise_word)

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


def assert_distribution(fillers_list, term_a, term_a_proportion, term_b, term_b_proportion, tolerance=0.25):
    term_a_sampled_proportion = len([filler for filler in fillers_list if term_a in filler]) / float(len(fillers_list))
    term_b_sampled_proportion = len([filler for filler in fillers_list if term_b in filler]) / float(len(fillers_list))
    assert(term_a_sampled_proportion > term_a_proportion - tolerance)
    assert(term_a_sampled_proportion < term_a_proportion + tolerance)
    assert(term_b_sampled_proportion > term_b_proportion - tolerance)
    assert(term_b_sampled_proportion < term_b_proportion + tolerance)

def test_train_set():
    with open(os.path.join(save_dir, 'train.p'), 'rb') as f:
        X, y = pickle.load(f)

    emcee_friend_answers = []
    subject_poet_answers = []
    for example_num in range(len(y)):
        input_sentence = np.array(np.squeeze(X[example_num]), dtype=int)
        query = wordlist[int(input_sentence[-1])]
        answer_word_index = int(y[example_num][0])
        answer = wordlist[answer_word_index]
        input_sentence = np.delete(np.squeeze(input_sentence), np.where(input_sentence == noise_index))
        if query in ['qemcee', 'qfriend']:
            emcee_friend_answers.append(answer)
        else:
            subject_poet_answers.append(answer)
        assert('_05_train' in answer)
        assert(int(input_sentence[query_to_input_indices[query][0]]) == answer_word_index)
    assert_distribution(subject_poet_answers, 'add', 0.75, 'subtract', 0.25)
    assert_distribution(emcee_friend_answers, 'subtract', 0.75, 'add', 0.25)

def test_test_set():
    with open(os.path.join(save_dir, 'test.p'), 'rb') as f:
        X, y = pickle.load(f)

    emcee_friend_answers = []
    subject_poet_answers = []
    for example_num in range(len(y)):
        input_sentence = np.array(np.squeeze(X[example_num]), dtype=int)
        query = wordlist[int(input_sentence[-1])]
        answer_word_index = int(y[example_num][0])
        answer = wordlist[answer_word_index]
        input_sentence = np.delete(np.squeeze(input_sentence), np.where(input_sentence == noise_index))
        if query in ['qemcee', 'qfriend']:
            emcee_friend_answers.append(answer)
        else:
            subject_poet_answers.append(answer)
        assert('_05_test' in answer)
        assert(int(input_sentence[query_to_input_indices[query][0]]) == answer_word_index)
        
    assert_distribution(subject_poet_answers, 'add', 0.75, 'subtract', 0.25)
    assert_distribution(emcee_friend_answers, 'subtract', 0.75, 'add', 0.25)


def test_flipped_test_set():
    with open(os.path.join(save_dir, 'test_flipped_distribution.p'), 'rb') as f:
        X, y = pickle.load(f)

    emcee_friend_answers = []
    subject_poet_answers = []
    for example_num in range(len(y)):
        input_sentence = np.array(np.squeeze(X[example_num]), dtype=int)
        query = wordlist[int(input_sentence[-1])]
        answer_word_index = int(y[example_num][0])
        answer = wordlist[answer_word_index]
        input_sentence = np.delete(np.squeeze(input_sentence), np.where(input_sentence == noise_index))
        if query in ['qemcee', 'qfriend']:
            emcee_friend_answers.append(answer)
        else:
            subject_poet_answers.append(answer)
        assert('_05_train' in answer)
        assert(int(input_sentence[query_to_input_indices[query][0]]) == answer_word_index)
        
    assert_distribution(subject_poet_answers, 'add', 0.25, 'subtract', 0.75)
    assert_distribution(emcee_friend_answers, 'subtract', 0.25, 'add', 0.75)



def test_unseen_flipped_test_set():
    with open(os.path.join(save_dir, 'test_unseen_flipped_distribution.p'), 'rb') as f:
        X, y = pickle.load(f)

    emcee_friend_answers = []
    subject_poet_answers = []
    for example_num in range(len(y)):
        input_sentence = np.array(np.squeeze(X[example_num]), dtype=int)
        query = wordlist[int(input_sentence[-1])]
        answer_word_index = int(y[example_num][0]) 
        answer = wordlist[answer_word_index]
        input_sentence = np.delete(np.squeeze(input_sentence), np.where(input_sentence == noise_index))
        if query in ['qemcee', 'qfriend']:
            emcee_friend_answers.append(answer)
        else:
            subject_poet_answers.append(answer)
        assert('_05_test' in answer)
        assert(int(input_sentence[query_to_input_indices[query][0]]) == answer_word_index)
        
    assert_distribution(subject_poet_answers, 'add', 0.25, 'subtract', 0.75)
    assert_distribution(emcee_friend_answers, 'subtract', 0.25, 'add', 0.75)


def test_unseen_no_addition_test_set():
    with open(os.path.join(save_dir, 'test_unseen_no_addition_distribution.p'), 'rb') as f:
        X, y = pickle.load(f)

    for example_num in range(len(y)):
        input_sentence = np.array(np.squeeze(X[example_num]), dtype=int)
        query = wordlist[int(input_sentence[-1])]
        answer_word_index = int(y[example_num][0]) 
        answer = wordlist[answer_word_index]
        input_sentence = np.delete(np.squeeze(input_sentence), np.where(input_sentence == noise_index))
        assert('add_05' not in answer)
        assert('subtract_05' not in answer)
        assert(int(input_sentence[query_to_input_indices[query][0]]) == answer_word_index)


def test_ambiguous_queried_role_test_set():
    with open(os.path.join(save_dir, 'test_ambiguous_queried_role.p'), 'rb') as f:
        X, y = pickle.load(f)

    for example_num in range(len(y)):
        input_sentence = np.array(np.squeeze(X[example_num]), dtype=int)
        query = wordlist[int(input_sentence[-1])]
        for input_index in query_to_input_indices[query]:
            assert((input_sentence[input_index] == noise_index) or (input_sentence[input_index + 1] == noise_index))


def test_ambiguous_all_test_set():
    with open(os.path.join(save_dir, 'test_ambiguous_all.p'), 'rb') as f:
        X, y = pickle.load(f)

    for example_num in range(len(y)):
        input_sentence = np.array(np.squeeze(X[example_num]), dtype=int)
        unique_input_words = np.unique(input_sentence[:-1])
        assert(len(unique_input_words) == 1)
        assert(noise_index in unique_input_words)
