"""Modules to create data for role-filler binding experiments."""
import argparse
import copy
import json
import numpy as np
import os
import pickle
import sys

sys.path.append("../")
sys.path.append("./")
import hard_coded_things

from embedding_util import create_word_vector
from directories import base_dir


def flatten_arrays(unflattened_list):
    return [item for sublist in unflattened_list for item in sublist]


def construct_all_state_sequences(transitions,
        start_frame=['begin'],
        end_state='end'):
    '''Returns a list of all paths from start to end state using transitions.

    Args:
    ====
    transitions: Dictionary mapping from state to outgoing states.
    start_frame: List of states that have been visited.
    end_state: Name of end state.

    Returns:
    =======
    state_sequences: A list of all state sequences from start to end state.
    '''
    state_sequences = []
    previous_state = start_frame[-1]
    if previous_state == end_state:
        return [start_frame]

    next_states = transitions[previous_state]
    for state in next_states:
        state_sequences += construct_all_state_sequences(transitions=transitions,
                start_frame=start_frame + [state],
                end_state=end_state)
    return state_sequences


def write_csw_experiment(experiment_name, num_examples_per_frame, num_unseen_examples_per_frame):
    print(experiment_name)
    """Create train and test sets for a role-filler binding experiment.

    Assumes story files have been written by Coffee Shop world.

    Args:
        experiment_name: Name of folder in which stories are stored.
                               Assumes stories are stored in the directory
                               home_dir + "narrative/story/", where home_dir is
                               defined in directories.py.

    Saves (in the directory base_dir + "data/experiment_name/", where
           base_dir is defined in directories.py):
        train.p: A pickle file containing:
                 X: [num_train_examples x num_words_per_story x 1] matrix of train inputs.
                 y: [num_train_examples x 1] matrix of correct train outputs.
        test.p: A pickle file containing:
                 X: [num_test_examples x num_words_per_story x 1] matrix of test inputs.
                 y: [num_test_examples x 1] matrix of correct test outputs.
        Xy_english.txt: A file containing human-readable versions of the inputs,
                        correct outputs, and the word list used in the experiment.
                        (Each X and y matrix represents words by their index in
                        the word list.)
    """
    experiment_name += "_AllQs"
    experiment_data_path = os.path.join(base_dir, "data", experiment_name)
    query_delimiter = "?"
    query_starter = "Q"
    padding_word = "zzz"
    distributions_dict = {
            "DESSERT": "A",
            "DRINK": "B",
            "EMCEE": "A",
            "FRIEND": "B",
            "POET": "A",
            "SUBJECT": "B",
            }
    if not os.path.exists(experiment_data_path):
        os.makedirs(experiment_data_path)

    # Create frames.
    with open('story_frame.json', 'r') as f:
        story_frame_info = json.load(f)
    transitions = story_frame_info['transitions']
    state_contents = story_frame_info['state_contents']
    role_types = story_frame_info['role_types']
    state_sequences = construct_all_state_sequences(transitions)
    assert(len(state_sequences) == 24)
    frames = [flatten_arrays([state_contents[state] for state in state_sequence]) for state_sequence in state_sequences]
    num_examples = len(frames) * num_examples_per_frame
    num_unseen_examples = len(frames) * num_unseen_examples_per_frame

    if 'variablefiller' in experiment_name:
        dummy_instances = {role: ['%sFILLER' % role] for role in role_types.keys()}
        train_instances, test_instances = dummy_instances, dummy_instances
    elif 'fixedfiller' in experiment_name:
        train_instances, test_instances = hard_coded_things.fixed_train_instances, hard_coded_things.fixed_test_instances

    query_choices = role_types.keys()
    wordslist = flatten_arrays(state_contents.values()) + flatten_arrays(train_instances.values()) + flatten_arrays(test_instances.values()) + [padding_word, query_delimiter]

    for query_choice in query_choices:
        wordslist.append(query_starter + query_choice)
    wordslist = list(set(wordslist))

    # Determine experiment information.
    max_story_length = max([len(frame) for frame in frames])
    input_dims = max_story_length + 3  # +2 for query delimiter and the actual query. +1 for padding at end.

    X = np.zeros([num_examples, input_dims, 1], dtype=np.int32)
    y = np.zeros([num_examples, 1], dtype=np.int32)
    test_unseen_X = np.zeros([num_unseen_examples, input_dims, 1], dtype=np.int32)
    test_unseen_y = np.zeros([num_unseen_examples, 1], dtype=np.int32)

    # Generate inputs and correct outputs from stories.
    for frame_index, frame in enumerate(frames):
        print('Generating for frame ', frame)
        padding_size = max_story_length - len(frame)
        frame_roles = [role for role in role_types.keys() if role in frame]
        for example_index in range(num_examples_per_frame):
            if example_index % 1000 == 0:
                print(example_index)
            story = copy.deepcopy(frame)
            role_assignments = {}
            for role in frame_roles:
                if 'fixedfiller' in experiment_name:
                    role_assignment = np.random.choice(train_instances[role_types[role]])
                    while role_assignment in role_assignments.values():
                        role_assignment = np.random.choice(train_instances[role_types[role]])
                elif 'variablefiller' in experiment_name:
                    role_assignment = '%sFILLER' % role
                role_assignments[role] = role_assignment
            story = [role_assignments[word] if word in role_assignments else word for word in story]
            queried_role = np.random.choice(list(role_assignments.keys()))
            query = query_starter + queried_role
            response = role_assignments[queried_role]

            # If necessary, add padding to end of story (ensures that inputs are all the same length).
            story += [padding_word] * (padding_size + 1)  # so we can shift all stories later.
            story += [query_delimiter, query]
            outputs = [response]

            # Convert to numerical representation and add to X and y.
            data_index = (num_examples_per_frame * frame_index) + example_index
            X[data_index, :, :] = np.expand_dims([wordslist.index(storyword) for storyword in story], axis=1)
            y[data_index, :] = [wordslist.index(output_word) for output_word in outputs]

        if 'fixedfiller' in experiment_name:
            for example_index in range(num_unseen_examples_per_frame):
                story = copy.deepcopy(frame)
                role_assignments = {}
                for role in frame_roles:
                    role_assignment = np.random.choice(test_instances[role_types[role]])
                    while role_assignment in role_assignments.values():
                        role_assignment = np.random.choice(test_instances[role_types[role]])
                    role_assignments[role] = role_assignment
                story = [role_assignments[word] if word in role_assignments else word for word in story]
                queried_role = np.random.choice(list(role_assignments.keys()))
                query = query_starter + queried_role
                response = role_assignments[queried_role]

                # If necessary, add padding to end of story (ensures that inputs are all the same length).
                story += [padding_word] * (padding_size + 1)  # so we can shift all stories later.
                story += [query_delimiter, query]
                outputs = [response]

                # Convert to numerical representation and add to X and y.
                data_index = (num_unseen_examples_per_frame * frame_index) + example_index
                test_unseen_X[data_index, :, :] = np.expand_dims([wordslist.index(storyword) for storyword in story], axis=1)
                test_unseen_y[data_index, :] = [wordslist.index(output_word) for output_word in outputs]

    # Assert no repeated stories.
    X, unique_seen_indices = np.unique(X, axis=0, return_index=True)
    y = y[unique_seen_indices]

    if 'fixedfiller' in experiment_name:
        num_train = int(4 * len(X)/ 5)
        train_indices = np.random.choice(len(X), num_train, replace=False)
        test_indices = np.array([idx for idx in range(len(X)) if idx not in train_indices])
        train_X = X[train_indices, :, :]
        train_y = y[train_indices, :]
        test_X = X[test_indices, :, :]
        test_y = y[test_indices, :]
        test_unseen_X, unique_unseen_indices = np.unique(test_unseen_X, axis=0, return_index=True)
        test_unseen_y = test_unseen_y[unique_unseen_indices]
    elif 'variablefiller' in experiment_name:
        train_X, train_y = X, y
        test_X, test_y = X, y
        test_unseen_X, test_unseen_y = X, y

    
    # Save data into pickle files.
    if not os.path.exists(experiment_data_path):
        os.makedirs(experiment_data_path)
    print(experiment_data_path)
    with open(os.path.join(experiment_data_path, 'train.p'), 'wb') as f:
        pickle.dump([train_X, train_y], f)
    with open(os.path.join(experiment_data_path, 'test.p'), 'wb') as f:
        pickle.dump([test_X, test_y], f)
    with open(os.path.join(experiment_data_path, 'test_unseen.p'), 'wb') as f:
        pickle.dump([test_unseen_X, test_unseen_y], f)
    with open(os.path.join(experiment_data_path, 'wordslist.p'), 'wb') as f:
        pickle.dump(wordslist, f)

    with open('../experiment_parameters.json', 'r') as f:
        experiment_parameters = json.load(f)

    experiment_parameters['input_dims'][experiment_name] = input_dims
    fillers = list(set(flatten_arrays(train_instances.values()) + flatten_arrays(test_instances.values())))
    experiment_parameters['filler_indices'][experiment_name] = [wordslist.index(filler) for filler in fillers]
    experiment_parameters['padding_indices'][experiment_name] = wordslist.index(padding_word)

    if 'variablefiller' in experiment_name:
        experiment_parameters['query_to_filler_index'][experiment_name] = {wordslist.index(query_starter + role): [wordslist.index(filler) for filler in dummy_instances[role]] for role in role_types.keys()}
        filler_distributions_dict = {wordslist.index(dummy_instances[role][0]): distributions_dict[role] for role in role_types.keys()}
        experiment_parameters['filler_distributions'][experiment_name] = filler_distributions_dict


    with open('../experiment_parameters.json', 'w') as f:
        json.dump(experiment_parameters, f)
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
    # Example use: python experiment_writer_rolefillerbinding.py --exp_name=poetrygeneralization_variablefiller_gensymbolicstates_100000_1 --gen_type=TESTONLY --poss_qs=Subject,Poet
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Name of the folder containing experiment stories', required=True)
    parser.add_argument('--num_examples_per_frame', help='Number of seen filler examples per frame', required=True, type=int)
    parser.add_argument('--num_unseen_examples_per_frame', help='Number of unseen filler examples per frame', required=True, type=int)
    args = parser.parse_args()
    write_csw_experiment(experiment_name=args.exp_name,
            num_examples_per_frame=args.num_examples_per_frame,
            num_unseen_examples_per_frame=args.num_unseen_examples_per_frame)
