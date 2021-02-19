import numpy as np
import os
import pickle
import sys
sys.path.append("../")
from directories import base_dir
from embedding_util import create_word_vector
"""
train3roles_testnewrole
"""
def generate_train3roles_testnewrole(num_persons_per_category, filler_distribution=None):
    NUM_DIMS = 50
    NUM_TRAIN_EXAMPLES = 24000
    NUM_TEST_EXAMPLES = 120
    NUM_UNSEEN_TEST_EXAMPLES = 120
    NUM_UNSEEN_FILLERS = 100
    SAVE_PATH = os.path.join("/", "home", "cc27", "Thesis", "generalized_schema_learning", "data", "generate_train3roles_testnewrole_withunseentestfillers_%dpersonspercategory_%dtrain_%dtest" % (num_persons_per_category, NUM_TRAIN_EXAMPLES, NUM_TEST_EXAMPLES))
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    STORY_FRAME = "begin subject sit subject friend announce emcee perform poet consume dessert drink goodbye".split(" ")
    QUESTIONS = ["QEmcee", "QFriend", "QPoet", "QSubject"]
    ROLES = ["emcee", "friend", "poet", "subject", "dessert", "drink"]
    PADDING_WORD = "zzz"
    NUM_PERSON_FILLERS = 4
    filler_indices = []
    for role in ROLES:
        filler_indices += list(np.where(np.array(STORY_FRAME) == role)[0])

    num_questions = len(QUESTIONS)
    person_fillers = [str(i) for i in range(num_persons_per_category * NUM_PERSON_FILLERS)]
    person_fillers_unseenintraining = [str(-1 * i) for i in range(1, NUM_UNSEEN_FILLERS + 1)]

    wordslist = list(set(STORY_FRAME + QUESTIONS + person_fillers + person_fillers_unseenintraining))
    wordslist.append(PADDING_WORD)
    story_frame_matrix = np.expand_dims(np.expand_dims(np.array([wordslist.index(word) for word in STORY_FRAME]), axis=1), axis=0)

    person_fillers_indices = [wordslist.index(filler) for filler in person_fillers]
    person_fillers_unseenintraining_indices = [wordslist.index(filler) for filler in person_fillers_unseenintraining]
    person_fillers_bytrainmissingrole = {}
    for i in range(NUM_PERSON_FILLERS):
        person_fillers_bytrainmissingrole[ROLES[i]] = person_fillers_indices[i * num_persons_per_category : (i + 1) * num_persons_per_category]

    person_fillers_bytrainrole = {}
    for i in range(NUM_PERSON_FILLERS):
        role = ROLES[i]
        person_fillers_bytrainrole[role] = np.array(list(set(person_fillers_indices) - set(person_fillers_bytrainmissingrole[role])))

    # Generate train data.
    PERSON_ROLES = ["emcee", "friend", "poet", "subject"]
    person_wordindices = {}
    for role in PERSON_ROLES:
        person_wordindices[role] = wordslist.index(role)

    person_storyindices = {}
    for role in PERSON_ROLES:
        person_storyindices[role] = np.where(np.squeeze(story_frame_matrix) == person_wordindices[role])[0]

    role_wordindices = {}
    for role in ROLES:
        role_wordindices[role] = wordslist.index(role)

    role_storyindices = {}
    for role in ROLES:
        role_storyindices[role] = np.where(np.squeeze(story_frame_matrix) == role_wordindices[role])[0]

    question_wordindices = [wordslist.index(question) for question in QUESTIONS]
    question_storyindices = {question:STORY_FRAME.index(role) for question, role in zip(question_wordindices, ROLES)}
    padding = np.reshape(np.array([wordslist.index(PADDING_WORD)]), (1, 1, 1))
    train_X = np.empty((0, story_frame_matrix.shape[1] + 2, story_frame_matrix.shape[2]))
    train_y = np.empty((0, 1))
    for i in range(NUM_TRAIN_EXAMPLES):
        story = np.copy(story_frame_matrix)
        for role in PERSON_ROLES:
            filler = np.random.choice(person_fillers_bytrainrole[role])
            story[0, person_storyindices[role], 0] = filler
        question = np.random.choice(question_wordindices)
        answer = [story.squeeze()[question_storyindices[question]]]
        story = np.concatenate((story, padding, np.reshape(question, (1, 1, 1))), axis=1)
        train_X = np.concatenate((train_X, story), axis=0)
        train_y = np.concatenate((train_y, np.reshape(np.array(answer), (1, 1))), axis=0)

    with open(os.path.join(SAVE_PATH, "train.p"), "wb") as f:
        pickle.dump([train_X, train_y], f)

    test_X = np.empty((0, story_frame_matrix.shape[1] + 2, story_frame_matrix.shape[2]))
    test_y = np.empty((0, 1))
    for i in range(NUM_TEST_EXAMPLES):
        story = np.copy(story_frame_matrix)
        for role in PERSON_ROLES:
            filler = np.random.choice(person_fillers_bytrainmissingrole[role])
            story[0, person_storyindices[role], 0] = filler
        question = np.random.choice(question_wordindices)
        answer = [story.squeeze()[question_storyindices[question]]]
        story = np.concatenate((story, padding, np.reshape(question, (1, 1, 1))), axis=1)
        test_X = np.concatenate((test_X, story), axis=0)
        test_y = np.concatenate((test_y, np.reshape(np.array(answer), (1, 1))), axis=0)


    with open(os.path.join(SAVE_PATH, "test.p"), "wb") as f:
        pickle.dump([test_X, test_y], f)

    for question in question_wordindices:
        split_testunseen_X = np.empty((0, story_frame_matrix.shape[1] + 2, story_frame_matrix.shape[2]))
        split_testunseen_y = np.empty((0, 1))
        for i in range(NUM_UNSEEN_TEST_EXAMPLES):
            story = np.copy(story_frame_matrix)
            for role in ROLES:
                filler = np.random.choice(person_fillers_unseenintraining_indices)
                story[0, role_storyindices[role], 0] = filler
            answer = [story.squeeze()[question_storyindices[question]]]
            story = np.concatenate((story, padding, np.reshape(question, (1, 1, 1))), axis=1)
            split_testunseen_X = np.concatenate((split_testunseen_X, story), axis=0)
            split_testunseen_y = np.concatenate((split_testunseen_y, np.reshape(np.array(answer), (1, 1))), axis=0)
        with open(os.path.join(SAVE_PATH, "test_%s_unseen.p" % wordslist[question]), "wb") as f:
            pickle.dump([split_testunseen_X, split_testunseen_y], f)
    
    embedding = []

    for i in range(len(wordslist)):
        word = wordslist[i]
        word_embedding = {}
        word_embedding['index'] = i
        word_embedding['word'] = word
        if word in person_fillers or word in person_fillers_unseenintraining:
            print(word, filler_distribution)
            word_embedding['vector'] = create_word_vector(filler_distribution)
        else:
           word_embedding['vector'] = create_word_vector()
        embedding.append(word_embedding)

    with open(os.path.join(SAVE_PATH, "embedding.p"), "wb") as f:
        pickle.dump(embedding, f)

    with open(os.path.join(SAVE_PATH, "wordslist.p"), "wb") as f:
        pickle.dump(wordslist, f)

def generate_train3roles_testnewrole_withunseentestfillers_shuffledtestset(story_frame, num_persons_per_category):
    NUM_DIMS = 50
    NUM_TRAIN_EXAMPLES = 24000
    NUM_TEST_EXAMPLES = 120
    NUM_UNSEEN_FILLERS = 100
    SAVE_PATH = os.path.join("/", "home", "cc27", "Thesis", "generalized_schema_learning", "data", "generate_train3roles_testnewrole_withunseentestfillers_%dpersonspercategory_%dtrain_%dtest" % (num_persons_per_category, NUM_TRAIN_EXAMPLES, NUM_TEST_EXAMPLES))
    SAVE_PATH_SHUFFLED = os.path.join("/", "home", "cc27", "Thesis", "generalized_schema_learning", "data", "generate_train3roles_testnewrole_withunseentestfillers_%dpersonspercategory_%dtrain_%dtest_shuffled" % (num_persons_per_category, NUM_TRAIN_EXAMPLES, NUM_TEST_EXAMPLES))
    
    # Copy embedding, wordlist, and train examples from original story.
    with open(os.path.join(SAVE_PATH, "embedding.p"), "rb") as f:
        embedding = pickle.load(f)

    with open(os.path.join(SAVE_PATH, "wordslist.p"), "rb") as f:
        wordslist = pickle.load(f)
    
    with open(os.path.join(SAVE_PATH, "train.p"), "rb") as f:
        train_X, train_y = pickle.load(f)

    with open(os.path.join(SAVE_PATH_SHUFFLED, "embedding.p"), "wb") as f:
        pickle.dump(embedding, f)

    with open(os.path.join(SAVE_PATH_SHUFFLED, "wordslist.p"), "wb") as f:
        pickle.dump(wordslist, f)
    
    with open(os.path.join(SAVE_PATH_SHUFFLED, "train.p"), "wb") as f:
        pickle.dump([train_X, train_y], f)

    # Get filler indices and fillers.
    #QUESTIONS = ["QEmcee", "QFriend", "QPoet", "QSubject", "QDessert", "QDrink"]
    QUESTIONS = ["QEmcee", "QFriend", "QPoet", "QSubject"]
    ROLES = ["emcee", "friend", "poet", "subject", "dessert", "drink"]
    PADDING_WORD = "zzz"
    NUM_PERSON_FILLERS = 4
    filler_indices = []
    for role in ROLES:
        filler_indices += list(np.where(np.array(story_frame) == role)[0])

    num_questions = len(QUESTIONS)
    person_fillers = [str(i) for i in range(num_persons_per_category * NUM_PERSON_FILLERS)]

    story_frame_matrix = np.expand_dims(np.expand_dims(np.array([wordslist.index(word) for word in story_frame]), axis=1), axis=0)

    person_fillers_indices = [wordslist.index(filler) for filler in person_fillers]
    person_fillers_bytrainmissingrole = {}
    for i in range(NUM_PERSON_FILLERS):
        person_fillers_bytrainmissingrole[ROLES[i]] = person_fillers_indices[i * num_persons_per_category : (i + 1) * num_persons_per_category]

    PERSON_ROLES = ["emcee", "friend", "poet", "subject"]
    person_wordindices = {}
    for role in PERSON_ROLES:
        person_wordindices[role] = wordslist.index(role)

    person_storyindices = {}
    for role in PERSON_ROLES:
        person_storyindices[role] = np.where(np.squeeze(story_frame_matrix) == person_wordindices[role])[0]

    role_wordindices = {}
    for role in ROLES:
        role_wordindices[role] = wordslist.index(role)

    role_storyindices = {}
    for role in ROLES:
        role_storyindices[role] = np.where(np.squeeze(story_frame_matrix) == role_wordindices[role])[0]

    question_wordindices = [wordslist.index(question) for question in QUESTIONS]
    question_storyindices = {question:story_frame.index(role) for question, role in zip(question_wordindices, ROLES)}
    padding = np.reshape(np.array([wordslist.index(PADDING_WORD)]), (1, 1, 1))
    
    #test_y = np.empty((0, 1))
    #test_X = np.empty((0, story_frame_matrix.shape[1] + 2, story_frame_matrix.shape[2]))
    ## Generate shuffled data with previously seen fillers in same roles.
    #for question in question_wordindices:
    #    split_test_X = np.empty((0, story_frame_matrix.shape[1] + 2, story_frame_matrix.shape[2]))
    #    split_test_y = np.empty((0, 1))
    #    for i in range(NUM_TEST_EXAMPLES):
    #        story = np.copy(story_frame_matrix)
    #        for role in PERSON_ROLES:
    #            filler = np.random.choice(person_fillers_bytrainmissingrole[role])
    #            story[0, person_storyindices[role], 0] = filler
    #        answer = [story.squeeze()[question_storyindices[question]]]
    #        story = np.concatenate((story, padding, np.reshape(question, (1, 1, 1))), axis=1)
    #        split_test_X = np.concatenate((split_test_X, story), axis=0)
    #        split_test_y = np.concatenate((split_test_y, np.reshape(np.array(answer), (1, 1))), axis=0)
    #    print(wordslist[question], np.unique(split_test_y))
    #    with open(os.path.join(SAVE_PATH_SHUFFLED, "test_%s.p" % (wordslist[question])), "wb") as f:
    #        pickle.dump([split_test_X, split_test_y], f)
    #    test_X = np.concatenate((test_X, split_test_X), axis=0)
    #    test_y = np.concatenate((test_y, split_test_y), axis=0)
    #with open(os.path.join(SAVE_PATH_SHUFFLED, "test.p"), "wb") as f:
    #        pickle.dump([test_X, test_y], f)

    test_X = np.empty((0, story_frame_matrix.shape[1] + 2, story_frame_matrix.shape[2]))
    test_y = np.empty((0, 1))
    for i in range(NUM_TEST_EXAMPLES):
        story = np.copy(story_frame_matrix)
        for role in PERSON_ROLES:
            filler = np.random.choice(person_fillers_bytrainmissingrole[role])
            story[0, person_storyindices[role], 0] = filler
        question = np.random.choice(question_wordindices)
        answer = [story.squeeze()[question_storyindices[question]]]
        story = np.concatenate((story, padding, np.reshape(question, (1, 1, 1))), axis=1)
        test_X = np.concatenate((test_X, story), axis=0)
        test_y = np.concatenate((test_y, np.reshape(np.array(answer), (1, 1))), axis=0)


    with open(os.path.join(SAVE_PATH_SHUFFLED, "test.p"), "wb") as f:
        pickle.dump([test_X, test_y], f)

def test_generated_data(num_persons_per_category=1000):
    NUM_TRAIN_EXAMPLES = 24000
    NUM_TEST_EXAMPLES = 120
    SAVE_PATH = os.path.join("/", "home", "cc27", "Thesis", "generalized_schema_learning", "data", "generate_train3roles_testnewrole_withunseentestfillers_%dpersonspercategory_%dtrain_%dtest" % (num_persons_per_category, NUM_TRAIN_EXAMPLES, NUM_TEST_EXAMPLES))
    SAVE_PATH_SHUFFLED = os.path.join("/", "home", "cc27", "Thesis", "generalized_schema_learning", "data", "generate_train3roles_testnewrole_withunseentestfillers_%dpersonspercategory_%dtrain_%dtest_shuffled" % (num_persons_per_category, NUM_TRAIN_EXAMPLES, NUM_TEST_EXAMPLES))

    # Test that fillers do not overlap btwn train and test set for each query.
    print("TESTING TRAIN VS TEST SETS")
    trainX, trainy = np.load(os.path.join(SAVE_PATH, "train.p"))
    testX, testy = np.load(os.path.join(SAVE_PATH, "test.p"))
    wordslist = np.load(os.path.join(SAVE_PATH, "wordslist.p"))
    questions = np.unique(trainX[:,-1,:])
    answers_dict = {}
    for question in questions:
        if wordslist[int(question)] in ['QSubject', 'QFriend', 'QEmcee', 'QPoet']:
            question_trainexamples = np.where(trainX[:,-1,0] == question)
            answer_trainexamples = np.unique(trainy[question_trainexamples])
            question_testexamples = np.where(testX[:,-1,0] == question)
            answer_testexamples = np.unique(testy[question_testexamples])
            answers_dict[question] = {"train":answer_trainexamples, "test":answer_testexamples}
            print('num train ex: ', len(answer_trainexamples), 'num test ex: ', len(answer_testexamples))
            assert(len(set(answer_trainexamples).intersection(answer_testexamples)) == 0)
    
    # Test that unseen test sets do not include any train fillers.
    keys = ["QSubject", "QEmcee", "QFriend", "QPoet"]
    print("TESTING UNSEEN TEST SETS")
    for key in keys:
        print(key)
        testunseenX, testunseeny = np.load(os.path.join(SAVE_PATH, "test_%s_unseen.p" % key))
        allanswers_test = np.unique(testy)
        allanswers_train = np.unique(trainy)
        allanswers_testunseen = np.unique(testunseeny)
        assert(len(set(allanswers_testunseen).intersection(set(allanswers_test))) == 0)
        assert(len(set(allanswers_testunseen).intersection(set(allanswers_train))) == 0)

    print("TESTING SHUFLED TEST SETS")
    trainX, trainy = np.load(os.path.join(SAVE_PATH_SHUFFLED, "train.p"))
    testX, testy = np.load(os.path.join(SAVE_PATH_SHUFFLED, "test.p"))
    testX_unshuffled, testy_unshuffled = np.load(os.path.join(SAVE_PATH, "test.p"))
    wordslist = np.load(os.path.join(SAVE_PATH_SHUFFLED, "wordslist.p"))
    questions = np.unique(trainX[:,-1,:])
    answers_dict = {}
    all_trainexamples = np.unique(trainy)
    for question in questions:
        if wordslist[int(question)] in ['QSubject', 'QFriend', 'QEmcee', 'QPoet']:
            question_trainexamples = np.where(trainX[:,-1,0] == question)
            answer_trainexamples = np.unique(trainy[question_trainexamples])
            question_testexamples = np.where(testX[:,-1,0] == question)
            answer_testexamples = np.unique(testy[question_testexamples])
            question_testexamples_unshuffled = np.where(testX_unshuffled[:,-1,0] == question)
            answer_testexamples_unshuffled = np.unique(testy_unshuffled[question_testexamples_unshuffled])
            answers_dict[question] = {"train":answer_trainexamples, "test":answer_testexamples}
            print('num train ex: ', len(answer_trainexamples), 'num test ex: ', len(answer_testexamples))
            assert(len(set(answer_trainexamples).intersection(answer_testexamples)) == 0)

if __name__ == '__main__':
    shuffled_frame = "consume dessert drink goodbye begin subject sit subject friend announce emcee perform poet".split(" ")
    generate_train3roles_testnewrole(1000, "randn")
    generate_train3roles_testnewrole_withunseentestfillers_shuffledtestset(story_frame=shuffled_frame, num_persons_per_category=1000)
    test_generated_data()
