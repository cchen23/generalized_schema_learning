"""Modules to create data for role-filler binding experiments."""
import argparse
import ast
import numpy as np
import os
import pickle
import re
import sys

sys.path.append("../")
sys.path.append("./")
from directories import home_dir, base_dir

def update_experimentdatapath(experiment_data_path, generalization_type):
    """Update experiment data path for specified generalization type."""
    if generalization_type == "TESTUNSEEN":
        experiment_data_path += "_testunseen"
    return experiment_data_path

def write_csw_experiment(experiment_foldername, generalization_type=None, possible_queries=None):
    print(experiment_foldername)
    print(generalization_type)
    print(possible_queries)
    """Create train and test sets for a role-filler binding experiment.

    Assumes story files have been written by Coffee Shop world.

    Args:
        experiment_foldername: Name of folder in which stories are stored.
                               Assumes stories are stored in the directory
                               home_dir + "narrative/story/", where home_dir is
                               defined in directories.py.
        generalization_type: Name of relationship describing filler name overlap
                             between train and test setself.
                             Options:
                                 NONE: Train and test sets use same fillers.
                                 TESTUNSEEN: Test fillers do not overlap with train fillers.
        possible_queries: List of roles that can be queried. If None, allows any
                          query for which all possible fillers consist of one word.

    Saves (in the directory base_dir + "data/experiment_foldername/", where
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
    story_data_path = os.path.join(home_dir, 'narrative', 'story')
    experiment_data_path = os.path.join(base_dir, "data", experiment_foldername)
    query_delimiter = "?"
    query_starter = "Q"
    padding_word = "zzz"
    test_unseen_generalization = generalization_type == "TESTUNSEEN"
    story_data_path = os.path.join(story_data_path, experiment_foldername)

    # Create directory to save experiment data.
    experiment_data_path = update_experimentdatapath(experiment_data_path, generalization_type)
    if possible_queries:
        path_extension = ""
        for query in possible_queries:
            path_extension += "Q" + query
        experiment_data_path += "_%s" % path_extension
    else:
        experiment_data_path += "_AllQs"
    if not os.path.exists(experiment_data_path):
        os.makedirs(experiment_data_path)

    # Get experiment info. Entities are types of fillers (e.g. "Person"), and roles
    # are parts of the story that are substituted by fillers (e.g. "Subject", "Friend").
    # Each role can be filled by one entity (e.g. "Subject" must be a "Person"),
    # and each entity can fill multiple roles (e.g. a "Person" can be a "Subject"
    # or a "Friend").
    experiment_foldername_split = experiment_foldername.split("_")
    QA_filename = "_".join(experiment_foldername_split[:-2]+["QA"]+experiment_foldername_split[-2:])
    entities_filename = "_".join(experiment_foldername_split[:-2]+["entities"]+experiment_foldername_split[-2:])
    storyfile = open(os.path.join(story_data_path, "%s.txt" % experiment_foldername), 'rb')
    stories = storyfile.read()
    wordslist = re.sub("[^\w]", " ", stories).split()
    wordslist = set(wordslist)
    storyfile.close()
    QA_file = open(os.path.join(story_data_path, "%s.txt" % QA_filename), 'r')
    entities_file = open(os.path.join(story_data_path, "%s.txt" % entities_filename), 'r')

    if test_unseen_generalization:
        train_instances = []
        test_instances = []

    # If necessary, set train and test instances.
    all_entities = ast.literal_eval(entities_file.readline()) # First line is a dict of all entities and their names.
    all_roles = ast.literal_eval(entities_file.readline()) # Second line is a dict of all roles and their names.

    # NOTE: Custom (hard-coded) train and test instances. Schema definitions ensure that the sets of train and test fillers (not just outputs) are disjoint.
    if 'variablefiller' in experiment_foldername:
        train_instances = ['PersonFillerTrain', 'FriendFillerTrain', 'EmceeFillerTrain', 'PoetFillerTrain', 'DrinkFillerTrain', 'DessertFillerTrain']
        test_instances = ['PersonFillerTest', 'FriendFillerTest', 'EmceeFillerTest', 'PoetFillerTest', 'DrinkFillerTest', 'DessertFillerTest']
    if 'fixedfiller' in experiment_foldername:
        train_instances = ['Mariko', 'Pradeep', 'Sarah', 'Julian', 'Jane', 'John', 'latte', 'water', 'juice', 'milk', 'espresso', 'chocolate', 'mousse', 'cookie', 'candy', 'cupcake', 'cheesecake', 'pastry']
        test_instances = ['Olivia', 'Will', 'Anna', 'Bill', 'coffee', 'tea', 'cake', 'sorbet']
    if test_unseen_generalization:
        print("Train instances:", train_instances)
        print("Unseen generalization instances:", test_instances)

    query_choices = all_roles.keys() # Assumes that all fillers have exactly one word.
    if possible_queries:
        query_choices = possible_queries
    print("Query choices: %s" % str(query_choices))

    # Update wordlist for padding and query words.
    for query_choice in query_choices:
        wordslist.add(query_starter + query_choice)
    wordslist.add(query_delimiter)
    wordslist.add(padding_word) # Padding used when stories are not all the same length.
    wordslist = list(wordslist)
    QA_file.close()

    # Determine experiment information.
    max_story_length = max([len(re.sub("[^\w]", " ", story).split()) for story in stories.split(' \n\n')])
    input_dims = max_story_length+(3) # +2 for query delimiter and the actual query. +1 for padding at end.
    num_classes = len(wordslist) # For one-hot encoding of each word.
    num_samples = int(experiment_foldername_split[-1]) * int(experiment_foldername_split[-2]) # Relies on naming convention used in narrative/.
    print("INPUT DIMS: %d" % input_dims)
    print("NUM CLASSES: %d" % num_classes)

    X = np.zeros([num_samples, input_dims, 1], dtype=np.int32)
    y = np.zeros([num_samples, 1], dtype=np.int32)
    if test_unseen_generalization:
        train_indices = []
        test_indices = []

    # Generate inputs and correct outputs from stories.
    story_file = open(os.path.join(story_data_path, "%s.txt" % experiment_foldername), 'rb')
    QA_file = open(os.path.join(story_data_path, "%s.txt" % (QA_filename)), 'r')
    verification_file = open(os.path.join(experiment_data_path, "Xy_english.txt"), 'wb')
    for i in range(num_samples):
        # Get story and info.
        story = story_file.readline()
        while story == '\n':
            story = story_file.readline()
        story = re.sub("[^\w]", " ", story).split(' \n')[0].split()
        entities = QA_file.readline()
        while entities[0] != "{":
            entities = QA_file.readline()
        attributes = QA_file.readline()
        entities = ast.literal_eval(entities)

        # Append a question about one entity.
        entity_query = np.random.choice(query_choices)
        entity_response = entities[entity_query]

        # Only allow possible queries (queries for which the role occurs in the story).
        while entity_response not in story:
            entity_query = np.random.choice(query_choices)
            entity_response = entities[entity_query]

        # If necessary, add padding to end of story (ensures that inputs are all the same length).
        padding_size = max_story_length - len(story)
        story += [padding_word] * (padding_size + 1) # so we can shift all stories later.
        story.append(query_delimiter)
        story.append("%s" % (query_starter + entity_query))
        outputs = ["%s" % entity_response]

        # Convert to numerical representation and add to X and y.
        X[i,:,:] = np.expand_dims([wordslist.index(storyword) for storyword in story], axis=1)
        y[i,:] = [wordslist.index(outputword) for outputword in outputs]

        # If necessary, separate into train and test sets based on fillers.
        if test_unseen_generalization:
            if len(set(story).intersection(set(train_instances))) == 0:
                test_indices.append(i)
            elif len(set(story).intersection(set(test_instances))) == 0:
                train_indices.append(i)

        # Write English-language version for sanity check.
        verification_file.write(" ".join(story) + '\n' + " ".join(outputs) + '\n' + "wordslist: " + str(wordslist) + '\n\n\n')

    # Remove repeated stories.
    print("********************************************************")
    print("X shape before removing non-unique stories:", X.shape)
    print("y shape before removing non-unique stories:", y.shape)
    X = np.squeeze(X)
    tempinputs = np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1])))
    _, idx = np.unique(tempinputs, return_index=True)
    X = X[idx]
    y = y[idx]
    if test_unseen_generalization:
        print("num train_indices: %d" % len(train_indices))
        print("num test_indices: %d" % len(test_indices))
    idx_list = idx.tolist()
    if test_unseen_generalization:
        print(len(idx))
        train_indices = list(set(idx_list).intersection(train_indices))
        test_indices = list(set(idx_list).intersection(test_indices))
        print("num train_indices: %d" % len(train_indices))
        print("num test_indices: %d" % len(test_indices))
        train_indices = [idx_list.index(train_index) for train_index in train_indices]
        test_indices = [idx_list.index(test_index) for test_index in test_indices]
    X = np.expand_dims(X, axis=-1)
    num_samples = X.shape[0] # NOTE: WILL NEED TO UPDATE THIS FOR ALTERNATING SCHEMAS
    print("NUM SAMPLES: %d" % num_samples)
    print("X shape after removing non-unique stories:", X.shape)
    print("y shape after removing non-unique stories:", y.shape)

    # Split X and y into train and test sets. If train and test instances
    # not specified, use 80/20 train/test split.
    if test_unseen_generalization:
        train_X = X[train_indices,:,:]
        train_y = y[train_indices,:]
        test_X = X[test_indices,:,:]
        test_y = y[test_indices,:]
    else:
        num_train = int(4*num_samples/5)
        train_X = X[:num_train,:,:]
        test_X = X[num_train:,:,:]
        train_y = y[:num_train,:]
        test_y = y[num_train:,:]

    # Save data into pickle files.
    if not os.path.exists(experiment_data_path):
        os.makedirs(experiment_data_path)
    print(experiment_data_path)
    with open(os.path.join(experiment_data_path, 'train.p'), 'wb') as f:
        pickle.dump([train_X, train_y], f)
    with open(os.path.join(experiment_data_path, 'test.p'), 'wb') as f:
        pickle.dump([test_X, test_y], f)

if __name__ == '__main__':
    # Example use: python experiment_writer_rolefillerbinding.py --exp_name=poetrygeneralization_variablefiller_gensymbolicstates_100000_1 --gen_type=TESTONLY --poss_qs=Subject,Poet
    parser=argparse.ArgumentParser()

    parser.add_argument('--exp_name', help='Name of the folder containing experiment stories', required=True)
    parser.add_argument('--gen_type', help='Optional: Generalization type (defaults to NONE)', choices=["NONE", "TESTUNSEEN"])
    parser.add_argument('--poss_qs', help='Optional: Possible queries. Enter separated by commas. Ex. Subject,Poet (defaults to all queries)', type=str)

    args=parser.parse_args()
    experiment_foldername = args.exp_name
    generalization_type = args.gen_type
    possible_queries = args.poss_qs.split(",") if args.poss_qs else None
    print("Generating data for CSW experiment %s" % experiment_foldername)
    write_csw_experiment(experiment_foldername, generalization_type, possible_queries)
