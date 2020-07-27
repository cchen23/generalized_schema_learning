import numpy as np
import os
import pickle
import sys
sys.path.append("../")
from directories import base_dir
from embedding_util import create_word_vector

def generate_probestatisticsretention_fixedfiller(percentage_train_indistribution=100, filler_distribution="add05", normalize_fillerdistribution=True):
    """
    NOTE: "outofdistribution", "indistribution", "middistribution" is according to special filler distribution.
    """
    print("********************************************************")
    print("generating probe statistics retention fixed filler with percentage_train_indistribution={percentage_train_indistribution}".format(percentage_train_indistribution=percentage_train_indistribution))
    NUM_PERSONS_PER_CATEGORY = 1000
    NUM_DIMS = 50
    NUM_TRAIN_EXAMPLES = 24000
    NUM_TEST_EXAMPLES = 120
    NUM_UNSEEN_TEST_EXAMPLES = 120
    SAVE_PATH = os.path.join("/", "home", "cc27", "Thesis", "generalized_schema_learning", "data", "probestatisticsretention_percentageindistribution{percentage_train_indistribution}_normalizefillerdistribution{normalize_fillerdistribution}".format(percentage_train_indistribution=percentage_train_indistribution, normalize_fillerdistribution=normalize_fillerdistribution))
    print("Saving to {save_path}".format(save_path=SAVE_PATH))
    STORY_FRAME = "begin subject sit subject friend announce emcee perform poet consume dessert drink goodbye".split(" ")
    QUESTIONS = ["QEmcee", "QFriend", "QPoet", "QSubject"]
    ROLES = ["emcee", "friend", "poet", "subject"]
    PADDING_WORD = "zzz"

    num_fillers = len(ROLES)

    # Get numerical representations of fillers.
    num_fillers_per_category = NUM_PERSONS_PER_CATEGORY * num_fillers
    fillers_indistribution = [i for i in range(num_fillers_per_category * 2)]
    fillers_training = fillers_indistribution[:int(num_fillers_per_category)]
    fillers_indistribution_unseen = fillers_indistribution[int(num_fillers_per_category):]
    fillers_outofdistribution_unseen = [-1 * i for i in range(1, num_fillers_per_category + 1)]
    fillers_middistribution_unseen = [i + np.max(fillers_indistribution) for i in range(num_fillers_per_category)]
    fillers_indistribution = [int(i) for i in fillers_indistribution]
    fillers_outofdistribution_unseen = [int(i) for i in fillers_outofdistribution_unseen]
    fillers_middistribution_unseen = [int(i) for i in fillers_middistribution_unseen]
    print("fillers_training", fillers_training)
    print("fillers_indistribution_unseen", fillers_indistribution_unseen)
    print("fillers_outofdistribution_unseen", fillers_outofdistribution_unseen)
    print("fillers_middistribution_indices_unseen", fillers_middistribution_unseen)

    # Get wordslist.
    wordslist = list(STORY_FRAME + QUESTIONS + fillers_indistribution + fillers_outofdistribution_unseen + fillers_middistribution_unseen)
    wordslist.append(PADDING_WORD)
    story_frame_matrix = np.expand_dims(np.expand_dims(np.array([wordslist.index(word) for word in STORY_FRAME]), axis=1), axis=0)
    padding = np.reshape(np.array([wordslist.index(PADDING_WORD)]), (1, 1, 1))

    # Get wordslist indices of fillers.
    fillers_indices_training = [wordslist.index(filler) for filler in fillers_training]
    fillers_indistribution_indices_unseen = [wordslist.index(filler) for filler in fillers_indistribution_unseen]
    fillers_outofdistribution_indices_unseen = [wordslist.index(filler) for filler in fillers_outofdistribution_unseen]
    fillers_middistribution_indices_unseen = [wordslist.index(filler) for filler in fillers_middistribution_unseen]

    # Get fillers used in each role during training.
    fillers_bytrainmissingrole = dict()
    for i in range(num_fillers):
        fillers_bytrainmissingrole[ROLES[i]] = fillers_indices_training[i * NUM_PERSONS_PER_CATEGORY : (i + 1) * NUM_PERSONS_PER_CATEGORY]

    fillers_bytrainrole = dict()
    for i in range(num_fillers):
        role = ROLES[i]
        fillers_bytrainrole[role] = np.array(list(set(fillers_indices_training) - set(fillers_bytrainmissingrole[role])))

    # Get indices of certain words in wordslist and in story.
    role_wordindices = dict()
    for role in ROLES:
        role_wordindices[role] = wordslist.index(role)

    role_storyindices = dict()
    for role in ROLES:
        role_storyindices[role] = np.where(np.squeeze(story_frame_matrix) == role_wordindices[role])[0]

    question_wordindices = [wordslist.index(question) for question in QUESTIONS]
    question_storyindices = {question:STORY_FRAME.index(role) for question, role in zip(question_wordindices, ROLES)}

    train_X = np.empty((0, story_frame_matrix.shape[1] + 2, story_frame_matrix.shape[2]))
    train_y = np.empty((0, 1))
    for i in range(NUM_TRAIN_EXAMPLES):
        story = np.copy(story_frame_matrix)
        for role in ROLES:
            filler = np.random.choice(fillers_bytrainrole[role])
            story[0, role_storyindices[role], 0] = filler
        question = np.random.choice(question_wordindices)
        answer = [story.squeeze()[question_storyindices[question]]]
        story = np.concatenate((story, padding, np.reshape(question, (1, 1, 1))), axis=1)
        train_X = np.concatenate((train_X, story), axis=0)
        train_y = np.concatenate((train_y, np.reshape(np.array(answer), (1, 1))), axis=0)
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    with open(os.path.join(SAVE_PATH, "train.p"), "wb") as f:
        pickle.dump([train_X, train_y], f)

    # Generate test set with excluded role-filler pairs.
    test_X = np.empty((0, story_frame_matrix.shape[1] + 2, story_frame_matrix.shape[2]))
    test_y = np.empty((0, 1))
    for i in range(NUM_TEST_EXAMPLES):
        story = np.copy(story_frame_matrix)
        for role in ROLES:
            filler = np.random.choice(fillers_bytrainmissingrole[role])
            story[0, role_storyindices[role], 0] = filler
        question = np.random.choice(question_wordindices)
        answer = [story.squeeze()[question_storyindices[question]]]
        story = np.concatenate((story, padding, np.reshape(question, (1, 1, 1))), axis=1)
        test_X = np.concatenate((test_X, story), axis=0)
        test_y = np.concatenate((test_y, np.reshape(np.array(answer), (1, 1))), axis=0)


    with open(os.path.join(SAVE_PATH, "test.p"), "wb") as f:
        pickle.dump([test_X, test_y], f)

    # Generate split test set with excluded role-filler pairs.
    for question in question_wordindices:
        split_test_X = np.empty((0, story_frame_matrix.shape[1] + 2, story_frame_matrix.shape[2]))
        split_test_y = np.empty((0, 1))
        for i in range(NUM_TEST_EXAMPLES):
            story = np.copy(story_frame_matrix)
            for role in ROLES:
                filler = np.random.choice(fillers_bytrainmissingrole[role])
                story[0, role_storyindices[role], 0] = filler
            answer = [story.squeeze()[question_storyindices[question]]]
            story = np.concatenate((story, padding, np.reshape(question, (1, 1, 1))), axis=1)
            split_test_X = np.concatenate((split_test_X, story), axis=0)
            split_test_y = np.concatenate((split_test_y, np.reshape(np.array(answer), (1, 1))), axis=0)
        print(wordslist[question], np.unique(split_test_y))
        with open(os.path.join(SAVE_PATH, "test_%s.p" % wordslist[question]), "wb") as f:
            pickle.dump([split_test_X, split_test_y], f)

    # Generate test set with unseen, in distribution fillers.
    for question in question_wordindices:
        split_testunseen_X = np.empty((0, story_frame_matrix.shape[1] + 2, story_frame_matrix.shape[2]))
        split_testunseen_y = np.empty((0, 1))
        for i in range(NUM_UNSEEN_TEST_EXAMPLES):
            story = np.copy(story_frame_matrix)
            for role in ROLES:
                filler = np.random.choice(fillers_indistribution_indices_unseen)
                story[0, role_storyindices[role], 0] = filler
            answer = [story.squeeze()[question_storyindices[question]]]
            story = np.concatenate((story, padding, np.reshape(question, (1, 1, 1))), axis=1)
            split_testunseen_X = np.concatenate((split_testunseen_X, story), axis=0)
            split_testunseen_y = np.concatenate((split_testunseen_y, np.reshape(np.array(answer), (1, 1))), axis=0)
        with open(os.path.join(SAVE_PATH, "test_%s_unseen_indistribution.p" % wordslist[question]), "wb") as f:
            pickle.dump([split_testunseen_X, split_testunseen_y], f)

    # Generate test set with unseen, out of distribution fillers.
    for question in question_wordindices:
        split_testunseen_X = np.empty((0, story_frame_matrix.shape[1] + 2, story_frame_matrix.shape[2]))
        split_testunseen_y = np.empty((0, 1))
        for i in range(NUM_UNSEEN_TEST_EXAMPLES):
            story = np.copy(story_frame_matrix)
            for role in ROLES:
                filler = np.random.choice(fillers_outofdistribution_indices_unseen)
                story[0, role_storyindices[role], 0] = filler
            answer = [story.squeeze()[question_storyindices[question]]]
            story = np.concatenate((story, padding, np.reshape(question, (1, 1, 1))), axis=1)
            split_testunseen_X = np.concatenate((split_testunseen_X, story), axis=0)
            split_testunseen_y = np.concatenate((split_testunseen_y, np.reshape(np.array(answer), (1, 1))), axis=0)
        with open(os.path.join(SAVE_PATH, "test_%s_unseen_outofdistribution.p" % wordslist[question]), "wb") as f:
            pickle.dump([split_testunseen_X, split_testunseen_y], f)

    # Generate test set with unseen, mid distribution fillers.
    for question in question_wordindices:
        split_testunseen_X = np.empty((0, story_frame_matrix.shape[1] + 2, story_frame_matrix.shape[2]))
        split_testunseen_y = np.empty((0, 1))
        for i in range(NUM_UNSEEN_TEST_EXAMPLES):
            story = np.copy(story_frame_matrix)
            for role in ROLES:
                filler = np.random.choice(fillers_middistribution_indices_unseen)
                story[0, role_storyindices[role], 0] = filler
            answer = [story.squeeze()[question_storyindices[question]]]
            story = np.concatenate((story, padding, np.reshape(question, (1, 1, 1))), axis=1)
            split_testunseen_X = np.concatenate((split_testunseen_X, story), axis=0)
            split_testunseen_y = np.concatenate((split_testunseen_y, np.reshape(np.array(answer), (1, 1))), axis=0)
        with open(os.path.join(SAVE_PATH, "test_%s_unseen_middistribution.p" % wordslist[question]), "wb") as f:
            pickle.dump([split_testunseen_X, split_testunseen_y], f)

    # Generate embedding.
    embedding = []
    fillers_training_indistribution = []
    fillers_training_outofdistribution = []
    for i in range(num_fillers):
        fillers_training_subset = fillers_training[i * NUM_PERSONS_PER_CATEGORY : (i + 1) * NUM_PERSONS_PER_CATEGORY]
        fillers_training_indistribution += fillers_training_subset[:int(NUM_PERSONS_PER_CATEGORY * percentage_train_indistribution / 100.0)]
        fillers_training_outofdistribution += fillers_training_subset[int(NUM_PERSONS_PER_CATEGORY * percentage_train_indistribution / 100.0):]
    print("fillers training indistribution", fillers_training_indistribution)
    print("fillers_training_outofdistribution", fillers_training_outofdistribution)
        
    for i in range(len(wordslist)):
        word = wordslist[i]
        word_embedding = {}
        word_embedding['index'] = i
        word_embedding['word'] = word
        if word in fillers_training_indistribution:
            print(word, "train filler, in distribution")
            word_embedding['vector'] = create_word_vector("add05", normalize_fillerdistribution=normalize_fillerdistribution)
        elif word in fillers_training_outofdistribution:
           print(word, "train filler, out of distribution")
           word_embedding['vector'] = create_word_vector()
        elif word in fillers_indistribution_unseen:
            print(word, "in distribution")
            word_embedding['vector'] = create_word_vector("add05", normalize_fillerdistribution=normalize_fillerdistribution)
        elif word in fillers_middistribution_unseen:
            print(word, "mid distribution")
            word_embedding['vector'] = create_word_vector("add025", normalize_fillerdistribution=normalize_fillerdistribution)
        else:
           word_embedding['vector'] = create_word_vector()
        embedding.append(word_embedding)

    with open(os.path.join(SAVE_PATH, "embedding.p"), "wb") as f:
        pickle.dump(embedding, f)

    with open(os.path.join(SAVE_PATH, "wordslist.p"), "wb") as f:
        pickle.dump(wordslist, f)

def generate_ambiguous_text_examples(percentage_train_indistribution, normalize_fillerdistribution=True):
    print("generating probe statistics retention fixed filler ambiguous test examples with percentage_train_indistribution={percentage_train_indistribution}".format(percentage_train_indistribution=percentage_train_indistribution))
    NUM_PERSONS_PER_CATEGORY = 1000
    NUM_TEST_EXAMPLES = 120
    SAVE_PATH = os.path.join("/", "home", "cc27", "Thesis", "generalized_schema_learning", "data", "probestatisticsretention_percentageindistribution{percentage_train_indistribution}_normalizefillerdistribution{normalize_fillerdistribution}".format(percentage_train_indistribution=percentage_train_indistribution, normalize_fillerdistribution=normalize_fillerdistribution))
    print("Saving to {save_path}".format(save_path=SAVE_PATH))
    STORY_FRAME = "begin subject sit subject friend announce emcee perform poet consume dessert drink goodbye".split(" ")
    QUESTIONS = ["QEmcee", "QFriend", "QPoet", "QSubject"]
    ROLES = ["emcee", "friend", "poet", "subject"]
    PADDING_WORD = "zzz"
    
    num_fillers = len(ROLES)
    
    with open(os.path.join(SAVE_PATH, "wordslist.p"), "rb") as f:
        wordslist = pickle.load(f)

    story_frame_matrix = np.expand_dims(np.expand_dims(np.array([wordslist.index(word) for word in STORY_FRAME]), axis=1), axis=0)
    PERSON_ROLES = ["emcee", "friend", "poet", "subject"]
    
    num_fillers_per_category = NUM_PERSONS_PER_CATEGORY * num_fillers
    fillers_indistribution = [i for i in range(num_fillers_per_category * 2)]
    fillers_training = fillers_indistribution[:int(num_fillers_per_category)]
    fillers_indices_training = [wordslist.index(filler) for filler in fillers_training]

    # Get fillers used in each role during training.
    fillers_bytrainmissingrole = dict()
    for i in range(num_fillers):
        fillers_bytrainmissingrole[ROLES[i]] = fillers_indices_training[i * NUM_PERSONS_PER_CATEGORY : (i + 1) * NUM_PERSONS_PER_CATEGORY]

    fillers_bytrainrole = dict()
    for i in range(num_fillers):
        role = ROLES[i]
        fillers_bytrainrole[role] = np.array(list(set(fillers_indices_training) - set(fillers_bytrainmissingrole[role])))
    
    person_wordindices = {}
    for role in PERSON_ROLES:
        person_wordindices[role] = wordslist.index(role)

    question_wordindices = [wordslist.index(question) for question in QUESTIONS]
    question_storyindices = {question:STORY_FRAME.index(role) for question, role in zip(question_wordindices, ROLES)}
    padding = np.reshape(np.array([wordslist.index(PADDING_WORD)]), (1, 1, 1))

    STORY_FRAME_REPLACE_SUBJECT = "begin zzz sit zzz friend announce emcee perform poet consume dessert drink goodbye".split(" ")
    STORY_FRAME_REPLACE_FRIEND = "begin subject sit subject zzz announce emcee perform poet consume dessert drink goodbye".split(" ")
    STORY_FRAME_REPLACE_EMCEE = "begin subject sit subject friend announce zzz perform poet consume dessert drink goodbye".split(" ")
    STORY_FRAME_REPLACE_POET = "begin subject sit subject friend announce emcee perform zzz consume dessert drink goodbye".split(" ")

    story_frame_manipulated_list = [STORY_FRAME_REPLACE_SUBJECT, STORY_FRAME_REPLACE_FRIEND, STORY_FRAME_REPLACE_EMCEE, STORY_FRAME_REPLACE_POET]
    savename_list = ["replacesubject", "replacefriend", "replaceemcee", "replacepoet"]
    # For each frame, fill in roles and create test set.
    for story_frame_manipulated, savename in zip(story_frame_manipulated_list, savename_list):
        story_frame_matrix_manipulated = np.expand_dims(np.expand_dims(np.array([wordslist.index(word) for word in story_frame_manipulated]), axis=1), axis=0)
        person_storyindices = {}
        for role in PERSON_ROLES:
            person_storyindices[role] = np.where(np.squeeze(story_frame_matrix_manipulated) == person_wordindices[role])[0]

        for question in question_wordindices:
            split_test_X = np.empty((0, story_frame_matrix_manipulated.shape[1] + 2, story_frame_matrix_manipulated.shape[2]))
            split_test_y = np.empty((0, 1))
            for i in range(NUM_TEST_EXAMPLES):
                story = np.copy(story_frame_matrix_manipulated)
                for role in PERSON_ROLES:
                    filler = np.random.choice(fillers_bytrainrole[role])
                    story[0, person_storyindices[role], 0] = filler
                answer = [story.squeeze()[question_storyindices[question]]]
                story = np.concatenate((story, padding, np.reshape(question, (1, 1, 1))), axis=1)
                split_test_X = np.concatenate((split_test_X, story), axis=0)
                split_test_y = np.concatenate((split_test_y, np.reshape(np.array(answer), (1, 1))), axis=0)
            print(wordslist[question], np.unique(split_test_y))
            with open(os.path.join(SAVE_PATH, "test_%s_%s.p" % (wordslist[question], savename)), "wb") as f:
                pickle.dump([split_test_X, split_test_y], f)

def generate_ambiguous_text_examples_allfiller(percentage_train_indistribution, normalize_fillerdistribution=True):
    print("generating probe statistics retention fixed filler ambiguous test examples with percentage_train_indistribution={percentage_train_indistribution}".format(percentage_train_indistribution=percentage_train_indistribution))
    NUM_PERSONS_PER_CATEGORY = 1000
    NUM_TEST_EXAMPLES = 120
    SAVE_PATH = os.path.join("/", "home", "cc27", "Thesis", "generalized_schema_learning", "data", "probestatisticsretention_percentageindistribution{percentage_train_indistribution}_normalizefillerdistribution{normalize_fillerdistribution}".format(percentage_train_indistribution=percentage_train_indistribution, normalize_fillerdistribution=normalize_fillerdistribution))
    print("Saving to {save_path}".format(save_path=SAVE_PATH))
    STORY_FRAME = "begin subject sit subject friend announce emcee perform poet consume dessert drink goodbye".split(" ")
    QUESTIONS = ["QEmcee", "QFriend", "QPoet", "QSubject"]
    ROLES = ["emcee", "friend", "poet", "subject"]
    PADDING_WORD = "zzz"
    
    num_fillers = len(ROLES)
    
    with open(os.path.join(SAVE_PATH, "wordslist.p"), "rb") as f:
        wordslist = pickle.load(f)

    story_frame_matrix = np.expand_dims(np.expand_dims(np.array([wordslist.index(word) for word in STORY_FRAME]), axis=1), axis=0)
    question_wordindices = [wordslist.index(question) for question in QUESTIONS]
    question_storyindices = {question:STORY_FRAME.index(role) for question, role in zip(question_wordindices, ROLES)}
    padding = np.reshape(np.array([wordslist.index(PADDING_WORD)]), (1, 1, 1))

    STORY_FRAME = "zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz".split(" ")

    story_frame = np.expand_dims(np.expand_dims(np.array([wordslist.index(word) for word in STORY_FRAME]), axis=1), axis=0)
    for question in question_wordindices:
        split_test_X = np.empty((0, story_frame.shape[1] + 2, story_frame.shape[2]))
        split_test_y = np.empty((0, 1))
        for i in range(NUM_TEST_EXAMPLES):
            story = np.copy(story_frame)
            answer = [story.squeeze()[question_storyindices[question]]]
            story = np.concatenate((story, padding, np.reshape(question, (1, 1, 1))), axis=1)
            split_test_X = np.concatenate((split_test_X, story), axis=0)
            split_test_y = np.concatenate((split_test_y, np.reshape(np.array(answer), (1, 1))), axis=0)
        print(wordslist[question], np.unique(split_test_y))
        with open(os.path.join(SAVE_PATH, "test_%s_%s.p" % (wordslist[question], "allfiller"+wordslist[question])), "wb") as f:
            pickle.dump([split_test_X, split_test_y], f)

if __name__ == '__main__':
    percentage_train_indistribution = int(sys.argv[1])
    normalize_fillerdistribution = sys.argv[2] == True
    print("percentage_train_indistribution", percentage_train_indistribution)
    print("normalize_fillerdistribution", normalize_fillerdistribution)
    generate_probestatisticsretention_fixedfiller(percentage_train_indistribution=percentage_train_indistribution, normalize_fillerdistribution=normalize_fillerdistribution)
    generate_ambiguous_text_examples(percentage_train_indistribution, normalize_fillerdistribution=normalize_fillerdistribution)
    generate_ambiguous_text_examples_allfiller(percentage_train_indistribution, normalize_fillerdistribution=normalize_fillerdistribution)
