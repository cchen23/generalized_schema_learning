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
    QUESTIONS = ["QEmcee", "QFriend", "QPoet", "QSubject", "QDessert", "QDrink"]
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

    for question in question_wordindices:
        split_test_X = np.empty((0, story_frame_matrix.shape[1] + 2, story_frame_matrix.shape[2]))
        split_test_y = np.empty((0, 1))
        for i in range(NUM_TEST_EXAMPLES):
            story = np.copy(story_frame_matrix)
            for role in PERSON_ROLES:
                filler = np.random.choice(person_fillers_bytrainmissingrole[role])
                story[0, person_storyindices[role], 0] = filler
            answer = [story.squeeze()[question_storyindices[question]]]
            story = np.concatenate((story, padding, np.reshape(question, (1, 1, 1))), axis=1)
            split_test_X = np.concatenate((split_test_X, story), axis=0)
            split_test_y = np.concatenate((split_test_y, np.reshape(np.array(answer), (1, 1))), axis=0)
        print(wordslist[question], np.unique(split_test_y))
        with open(os.path.join(SAVE_PATH, "test_%s.p" % wordslist[question]), "wb") as f:
            pickle.dump([split_test_X, split_test_y], f)

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
    NUM_UNSEEN_TEST_EXAMPLES = 120
    NUM_UNSEEN_FILLERS = 100
    SAVE_PATH = os.path.join("/", "home", "cc27", "Thesis", "generalized_schema_learning", "data", "generate_train3roles_testnewrole_withunseentestfillers_%dpersonspercategory_%dtrain_%dtest" % (num_persons_per_category, NUM_TRAIN_EXAMPLES, NUM_TEST_EXAMPLES))

    with open(os.path.join(SAVE_PATH, "embedding.p"), "rb") as f:
        embedding = pickle.load(f)

    with open(os.path.join(SAVE_PATH, "wordslist.p"), "rb") as f:
        wordslist = pickle.load(f)

    QUESTIONS = ["QEmcee", "QFriend", "QPoet", "QSubject", "QDessert", "QDrink"]
    ROLES = ["emcee", "friend", "poet", "subject", "dessert", "drink"]
    PADDING_WORD = "zzz"
    NUM_PERSON_FILLERS = 4
    filler_indices = []
    for role in ROLES:
        filler_indices += list(np.where(np.array(story_frame) == role)[0])

    num_questions = len(QUESTIONS)
    person_fillers = [str(i) for i in range(num_persons_per_category * NUM_PERSON_FILLERS)]
    person_fillers_unseenintraining = [str(-1 * i) for i in range(1, NUM_UNSEEN_FILLERS + 1)]

    story_frame_matrix = np.expand_dims(np.expand_dims(np.array([wordslist.index(word) for word in story_frame]), axis=1), axis=0)

    person_fillers_indices = [wordslist.index(filler) for filler in person_fillers]
    person_fillers_unseenintraining_indices = [wordslist.index(filler) for filler in person_fillers_unseenintraining]
    person_fillers_bytrainmissingrole = {}
    for i in range(NUM_PERSON_FILLERS):
        person_fillers_bytrainmissingrole[ROLES[i]] = person_fillers_indices[i * num_persons_per_category : (i + 1) * num_persons_per_category]

    person_fillers_bytrainrole = {}
    for i in range(NUM_PERSON_FILLERS):
        role = ROLES[i]
        person_fillers_bytrainrole[role] = np.array(list(set(person_fillers_indices) - set(person_fillers_bytrainmissingrole[role])))

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

    # Generate shuffled data with previously seen fillers in same roles.
    for question in question_wordindices:
        split_test_X = np.empty((0, story_frame_matrix.shape[1] + 2, story_frame_matrix.shape[2]))
        split_test_y = np.empty((0, 1))
        for i in range(NUM_TEST_EXAMPLES):
            story = np.copy(story_frame_matrix)
            for role in PERSON_ROLES:
                filler = np.random.choice(person_fillers_bytrainrole[role])
                story[0, person_storyindices[role], 0] = filler
            answer = [story.squeeze()[question_storyindices[question]]]
            story = np.concatenate((story, padding, np.reshape(question, (1, 1, 1))), axis=1)
            split_test_X = np.concatenate((split_test_X, story), axis=0)
            split_test_y = np.concatenate((split_test_y, np.reshape(np.array(answer), (1, 1))), axis=0)
        print(wordslist[question], np.unique(split_test_y))
        with open(os.path.join(SAVE_PATH, "test_%s_shuffled_sameroles.p" % (wordslist[question])), "wb") as f:
            pickle.dump([split_test_X, split_test_y], f)

    # Generate shuffled data with previously seen fillers in different roles.
    for question in question_wordindices:
        split_test_X = np.empty((0, story_frame_matrix.shape[1] + 2, story_frame_matrix.shape[2]))
        split_test_y = np.empty((0, 1))
        for i in range(NUM_TEST_EXAMPLES):
            story = np.copy(story_frame_matrix)
            for role in PERSON_ROLES:
                filler = np.random.choice(person_fillers_bytrainmissingrole[role])
                story[0, person_storyindices[role], 0] = filler
            answer = [story.squeeze()[question_storyindices[question]]]
            story = np.concatenate((story, padding, np.reshape(question, (1, 1, 1))), axis=1)
            split_test_X = np.concatenate((split_test_X, story), axis=0)
            split_test_y = np.concatenate((split_test_y, np.reshape(np.array(answer), (1, 1))), axis=0)
        print(wordslist[question], np.unique(split_test_y))
        with open(os.path.join(SAVE_PATH, "test_%s_shuffled_differentroles.p" % (wordslist[question])), "wb") as f:
            pickle.dump([split_test_X, split_test_y], f)

    # Generate shuffled data with previously unseen fillers.
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
        with open(os.path.join(SAVE_PATH, "test_%s_shuffled_unseen.p" % (wordslist[question])), "wb") as f:
            pickle.dump([split_testunseen_X, split_testunseen_y], f)

def test_generated_data():
    print("TESTING TRAIN VS TEST SETS")
    trainX, trainy = np.load("train.p")
    testX, testy = np.load("test.p")
    questions = np.unique(trainX[:,-1,:])
    answers_dict = {}
    for question in questions:
        question_trainexamples = np.where(trainX[:,-1,0] == question)
        answer_trainexamples = np.unique(trainy[question_trainexamples])
        question_testexamples = np.where(testX[:,-1,0] == question)
        answer_testexamples = np.unique(testy[question_testexamples])
        answers_dict[question] = {"train":answer_trainexamples, "test":answer_testexamples}
        print(len(answer_trainexamples), len(answer_testexamples))
        print(set(answer_trainexamples).intersection(answer_testexamples))

    keys = ["QSubject", "QDessert", "QEmcee", "QFriend", "QDrink", "QPoet"]
    print("TESTING UNSEEN TEST SETS")
    for key in keys:
        print(key)
        testunseenX, testunseeny = np.load("test_%s_unseen.p" % key)
        allanswers_test = np.unique(testy)
        allanswers_train = np.unique(trainy)
        allanswers_testunseen = np.unique(testunseeny)
        print(set(allanswers_testunseen).intersection(set(allanswers_test)))
        print(set(allanswers_testunseen).intersection(set(allanswers_train)))

    print("TESTING SPLIT TEST SETS")
    for key in keys:
        print(key)
        splitX, splity = np.load("test_%s.p" % key)
        question = np.unique(splitX[:,-1,0])[0]
        print("unique questions: ", question)
        split_answers = np.unique(splity)
        answer_trainexamples = answers_dict[question]["train"]
        answer_testexamples = answers_dict[question]["test"]
        print("split answers length", len(split_answers), "train answers length", len(answer_trainexamples), "test answers length", len(answer_testexamples))
        print(set(split_answers).intersection(answer_testexamples))
        print(set(split_answers).intersection(answer_trainexamples))

    print("TESTING SHUFFLED1 TEST SETS")
    with open("wordslist.p", "rb") as f:
        wordslist = pickle.load(f)

    for key in keys:
        for description in ["differentroles", "sameroles", "unseen"]:
            print(key, description)
            splitX, splity = np.load("test_%s_shuffled_%s.p" % (key, description))
            question = np.unique(splitX[:,-1,0])[0]
            print("unique questions: ", question)
            split_answers = np.unique(splity)
            answer_trainexamples = answers_dict[question]["train"]
            answer_testexamples = answers_dict[question]["test"]
            print("split answers length", len(split_answers), "train answers length", len(answer_trainexamples), "test answers length", len(answer_testexamples))
            print(set(split_answers).intersection(answer_testexamples))
            print(set(split_answers).intersection(answer_trainexamples))
            index = np.random.choice(splitX.shape[0])
            sentence = [wordslist[i] for i in np.squeeze(splitX[index,:,:]).astype(int)]
            print(sentence)

if __name__ == '__main__':
    shuffled_frame = "consume dessert drink goodbye begin subject sit subject friend announce emcee perform poet".split(" ")
    generate_train3roles_testnewrole(1000, "randn")
    generate_train3roles_testnewrole_withunseentestfillers_shuffledtestset(story_frame=shuffled_frame, num_persons_per_category=1000)
    test_generated_data()
