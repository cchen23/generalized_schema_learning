import numpy as np
import os
import pickle
import sys
sys.path.append("../")
from embedding_util import create_word_vector

np.random.seed(123)


def get_question_for_role(role):
    return "q" + role


def write_examples(fillers_by_role_dict,
                   story_frame_matrix,
                   num_examples,
                   roles,
                   role_story_indices,
                   question_wordlist_indices,
                   noise_wordlist_index,
                   save_path,
                   ambiguous=None):
    X = np.empty((0, story_frame_matrix.shape[1] + 2, story_frame_matrix.shape[2]))
    y = np.empty((0, 1))
    noise_matrix = np.reshape(np.array([noise_wordlist_index]), (1, 1, 1))
    story_length = story_frame_matrix.shape[1]

    if ambiguous == 'all':
        for role in roles:
            story = np.copy(story_frame_matrix)
            story[0, :] = noise_wordlist_index
            story = np.concatenate((story, noise_matrix, np.reshape(question_wordlist_indices[role], (1, 1, 1))), axis=1)
            answer = noise_wordlist_index
            X = np.concatenate((X, story), axis=0)
            y = np.concatenate((y, np.reshape(np.array(answer), (1, 1))), axis=0)
    else:
        for i in range(num_examples):
            story = np.copy(story_frame_matrix)
            for role in roles:
                filler = np.random.choice(fillers_by_role_dict[role])
                story[0, role_story_indices[role], 0] = filler
            queried_role = np.random.choice(roles)
            answer = [story.squeeze()[role_story_indices[queried_role][0]]]
            if ambiguous == 'queried_role':
                story[0, role_story_indices[queried_role]] = noise_wordlist_index
            story_with_noise = np.concatenate((story, noise_matrix), axis=1)
            story = np.concatenate((story_with_noise, np.reshape(question_wordlist_indices[queried_role], (1, 1, 1))), axis=1)
            X = np.concatenate((X, story), axis=0)
            y = np.concatenate((y, np.reshape(np.array(answer), (1, 1))), axis=0)

    with open(save_path, "wb") as f:
        pickle.dump([X, y], f)


def generate_experiments(num_dims=50,
                         num_train_examples=12000,
                         num_test_examples=120,
                         num_train_fillers_per_category=1000,
                         num_test_fillers_per_category=1000,
                         normalize_filler_distribution=False,
                         save_dir=os.path.join("/", "home", "cc27", "Thesis", "generalized_schema_learning", "data", "probe_role_statistic_recall")):
    print("Saving to {save_dir}".format(save_dir=save_dir))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    STORY_FRAME = "begin subject sit subject friend announce emcee perform poet consume dessert drink goodbye".split(" ")
    ROLES = ["emcee", "friend", "poet", "subject"]
    NOISE_WORD = "zzz"

    questions = [get_question_for_role(role) for role in ROLES]

    # Get wordlist.
    add_05_fillers_train = ['add_05_train' + str(filler_index) for filler_index in range(num_train_fillers_per_category)]
    add_05_fillers_test = ['add_05_test' + str(filler_index) for filler_index in range(num_test_fillers_per_category)]
    add_05_fillers = add_05_fillers_train + add_05_fillers_test
    
    subtract_05_fillers_train = ['subtract_05_train' + str(filler_index) for filler_index in range(num_train_fillers_per_category)]
    subtract_05_fillers_test = ['subtract_05_test' + str(filler_index) for filler_index in range(num_test_fillers_per_category)]
    subtract_05_fillers = subtract_05_fillers_train + subtract_05_fillers_test
    no_addition_fillers = [str(filler_index) for filler_index in range(num_test_fillers_per_category)]
    fillers = add_05_fillers + subtract_05_fillers + no_addition_fillers
    wordlist = list(STORY_FRAME + questions + fillers)
    wordlist.append(NOISE_WORD)
    story_frame_matrix = np.expand_dims(np.expand_dims(np.array([wordlist.index(word) for word in STORY_FRAME]), axis=1), axis=0)

    # Get indices of words.
    role_wordlist_indices = dict()
    for role in ROLES:
        role_wordlist_indices[role] = wordlist.index(role)

    role_story_indices = dict()
    for role in ROLES:
        role_story_indices[role] = np.where(np.array(STORY_FRAME) == role)[0]

    question_wordlist_indices = {role: wordlist.index(get_question_for_role(role)) for role in ROLES}
    noise_wordlist_index = wordlist.index(NOISE_WORD)
    add_05_fillers_indices_train = [wordlist.index(filler) for filler in add_05_fillers_train]
    subtract_05_fillers_indices_train = [wordlist.index(filler) for filler in subtract_05_fillers_train]
    add_05_fillers_indices_test = [wordlist.index(filler) for filler in add_05_fillers_test]
    subtract_05_fillers_indices_test = [wordlist.index(filler) for filler in subtract_05_fillers_test]
    no_addition_fillers_indices = [wordlist.index(filler) for filler in no_addition_fillers]

    # Generate training set.
    train_fillers_by_role_dict = {
            'emcee': subtract_05_fillers_indices_train,
            'friend': subtract_05_fillers_indices_train,
            'poet': add_05_fillers_indices_train,
            'subject': add_05_fillers_indices_train,
            }
    write_examples(fillers_by_role_dict=train_fillers_by_role_dict,
                   story_frame_matrix=story_frame_matrix,
                   num_examples=num_train_examples,
                   roles=ROLES,
                   role_story_indices=role_story_indices,
                   question_wordlist_indices=question_wordlist_indices,
                   noise_wordlist_index=noise_wordlist_index,
                   save_path=os.path.join(save_dir, "train.p"))

    # Generate in distribution test set.
    test_fillers_by_role_dict = {
            'emcee': subtract_05_fillers_indices_test,
            'friend': subtract_05_fillers_indices_test,
            'poet': add_05_fillers_indices_test,
            'subject': add_05_fillers_indices_test,
            }
    write_examples(fillers_by_role_dict=test_fillers_by_role_dict,
                   story_frame_matrix=story_frame_matrix,
                   num_examples=num_test_examples,
                   roles=ROLES,
                   role_story_indices=role_story_indices,
                   question_wordlist_indices=question_wordlist_indices,
                   noise_wordlist_index=noise_wordlist_index,
                   save_path=os.path.join(save_dir, "test.p"))

    # Generate out of distribution test set. (draw each filler from the opposite test pool)
    test_flipped_distribution_fillers_by_role_dict = {
            'emcee': add_05_fillers_indices_train,
            'friend': add_05_fillers_indices_train,
            'poet': subtract_05_fillers_indices_train,
            'subject': subtract_05_fillers_indices_train,
            }
    write_examples(fillers_by_role_dict=test_flipped_distribution_fillers_by_role_dict,
                   story_frame_matrix=story_frame_matrix,
                   num_examples=num_test_examples,
                   roles=ROLES,
                   role_story_indices=role_story_indices,
                   question_wordlist_indices=question_wordlist_indices,
                   noise_wordlist_index=noise_wordlist_index,
                   save_path=os.path.join(save_dir, "test_flipped_distribution.p"))

    # Generate unseen flipped_distribution test set.
    unseen_flipped_distribution_fillers_by_role_dict = {
            'emcee': add_05_fillers_indices_test,
            'friend': add_05_fillers_indices_test,
            'poet': subtract_05_fillers_indices_test,
            'subject': subtract_05_fillers_indices_test,
            }
    write_examples(fillers_by_role_dict=unseen_flipped_distribution_fillers_by_role_dict,
                   story_frame_matrix=story_frame_matrix,
                   num_examples=num_test_examples,
                   roles=ROLES,
                   role_story_indices=role_story_indices,
                   question_wordlist_indices=question_wordlist_indices,
                   noise_wordlist_index=noise_wordlist_index,
                   save_path=os.path.join(save_dir, "test_unseen_flipped_distribution.p"))

    # Generate unseen ood test set.
    unseen_no_addition_fillers_by_role_dict = {
            'emcee': no_addition_fillers_indices,
            'friend': no_addition_fillers_indices,
            'poet': no_addition_fillers_indices,
            'subject': no_addition_fillers_indices,
            }
    write_examples(fillers_by_role_dict=unseen_no_addition_fillers_by_role_dict,
                   story_frame_matrix=story_frame_matrix,
                   num_examples=num_test_examples,
                   roles=ROLES,
                   role_story_indices=role_story_indices,
                   question_wordlist_indices=question_wordlist_indices,
                   noise_wordlist_index=noise_wordlist_index,
                   save_path=os.path.join(save_dir, "test_unseen_no_addition_distribution.p"))

    # Generate ambiguous test set. (replace the queried filler with padding)
    write_examples(fillers_by_role_dict=train_fillers_by_role_dict,
                   story_frame_matrix=story_frame_matrix,
                   num_examples=num_test_examples,
                   roles=ROLES,
                   role_story_indices=role_story_indices,
                   question_wordlist_indices=question_wordlist_indices,
                   noise_wordlist_index=noise_wordlist_index,
                   save_path=os.path.join(save_dir, "test_ambiguous_queried_role.p"),
                   ambiguous='queried_role')

    # Generate fully ambiguous test set. (replace the entire story frame with padding)
    write_examples(fillers_by_role_dict=train_fillers_by_role_dict,
                   story_frame_matrix=story_frame_matrix,
                   num_examples=num_test_examples,
                   roles=ROLES,
                   role_story_indices=role_story_indices,
                   question_wordlist_indices=question_wordlist_indices,
                   noise_wordlist_index=noise_wordlist_index,
                   save_path=os.path.join(save_dir, "test_ambiguous_all.p"),
                   ambiguous='all')

    # Generate embedding.
    embedding = []

    for i in range(len(wordlist)):
        word = wordlist[i]
        word_embedding = {}
        word_embedding['index'] = i
        word_embedding['word'] = word
        if word in add_05_fillers:
            word_embedding['vector'] = create_word_vector("add05", normalize_filler_distribution=normalize_filler_distribution)
        elif word in subtract_05_fillers:
            word_embedding['vector'] = create_word_vector("subtract05", normalize_filler_distribution=normalize_filler_distribution)
        else:
            word_embedding['vector'] = create_word_vector()
        embedding.append(word_embedding)

    with open(os.path.join(save_dir, "embedding.p"), "wb") as f:
        pickle.dump(embedding, f)

    with open(os.path.join(save_dir, "wordlist.p"), "wb") as f:
        pickle.dump(wordlist, f)


if __name__ == '__main__':
    generate_experiments()
