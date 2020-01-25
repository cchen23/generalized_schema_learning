import numpy as np
import os
import pickle
import sys
sys.path.append("../")
from directories import base_dir
from embedding_util import create_word_vector
"""
Fixed order, each train filler seen once.
"""
def generate_onefillerperrole():
    NUM_TRAIN_EXAMPLES = 24000
    NUM_TEST_EXAMPLES = 120
    NUM_DIMS = 50
    SAVE_PATH = os.path.join("/", "home", "cc27", "Thesis", "generalized_schema_learning", "data", "storyv2_train20000_AllQs")

    STORY_FRAME = "begin subject sit subject friend announce emcee perform poet consume dessert drink goodbye".split(" ")
    QUESTIONS = ["QEmcee", "QFriend", "QPoet", "QSubject", "QDessert", "QDrink"]
    ROLES = ["emcee", "friend", "poet", "subject", "dessert", "drink"]
    PADDING_WORD = "zzz"
    filler_indices = []
    for role in ROLES:
        filler_indices += list(np.where(np.array(STORY_FRAME) == role)[0])
    num_questions = len(QUESTIONS)
    wordslist = list(set(STORY_FRAME + QUESTIONS))
    wordslist.append(PADDING_WORD)
    story_frame_matrix = np.expand_dims(np.expand_dims(np.array([wordslist.index(word) for word in STORY_FRAME]), axis=1), axis=0)

    def generate_data(num_examples, questions, roles):
        num_questions = len(questions)
        stories = np.repeat(story_frame_matrix, num_questions, axis=0)
        padding = np.reshape(np.repeat([wordslist.index(PADDING_WORD)], num_questions), (num_questions, 1, 1))
        queries = np.reshape(np.array([wordslist.index(question) for question in questions]), (num_questions, 1, 1))
        stories = np.concatenate((stories, padding, queries), axis=1)
        answers = np.reshape(np.array([wordslist.index(role) for role in roles]), (num_questions, 1))
        num_repeats = num_examples // num_questions
        stories = np.repeat(stories, num_repeats, axis=0)
        answers = np.repeat(answers, num_repeats, axis=0)
        return stories, answers

    train_X, train_y = generate_data(NUM_TRAIN_EXAMPLES, QUESTIONS, ROLES)
    test_X, test_y = generate_data(NUM_TEST_EXAMPLES, QUESTIONS, ROLES)
    with open(os.path.join(SAVE_PATH, "train.p"), "wb") as f:
        pickle.dump([train_X, train_y], f)

    with open(os.path.join(SAVE_PATH, "test.p"), "wb") as f:
        pickle.dump([test_X, test_y], f)

    for question, role in zip(QUESTIONS, ROLES):
        split_test_X, split_test_y = generate_data(NUM_TEST_EXAMPLES, [question], [role])
        with open(os.path.join(SAVE_PATH, "test_%s.p" % question), "wb") as f:
            pickle.dump([split_test_X, split_test_y], f)

    import sys
    sys.path.append("../")
    from directories import base_dir
    from embedding_util import create_word_vector

    embedding = []

    for i in range(len(wordslist)):
        word = wordslist[i]
        word_embedding = {}
        word_embedding['index'] = i
        word_embedding['word'] = word
        word_embedding['vector'] = create_word_vector()
        embedding.append(word_embedding)

    with open(os.path.join(SAVE_PATH, "embedding.p"), "wb") as f:
        pickle.dump(embedding, f)

    with open(os.path.join(SAVE_PATH, "wordslist.p"), "wb") as f:
        pickle.dump(wordslist, f)

"""
train3roles_testnewrole
"""
def generate_train3roles_testnewrole(num_persons_per_category):
    NUM_DIMS = 50
    NUM_TRAIN_EXAMPLES = 24000
    NUM_TEST_EXAMPLES = 120
    NUM_UNSEEN_TEST_EXAMPLES = 120
    NUM_UNSEEN_FILLERS = 100
    SAVE_PATH = os.path.join("/", "home", "cc27", "Thesis", "generalized_schema_learning", "data", "generate_train3roles_testnewrole_withunseentestfillers_%dpersonspercategory_%dtrain_%dtest" % (num_persons_per_category, NUM_TRAIN_EXAMPLES, NUM_TEST_EXAMPLES))

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
        word_embedding['vector'] = create_word_vector()
        embedding.append(word_embedding)

    with open(os.path.join(SAVE_PATH, "embedding.p"), "wb") as f:
        pickle.dump(embedding, f)

    with open(os.path.join(SAVE_PATH, "wordslist.p"), "wb") as f:
        pickle.dump(wordslist, f)

def generate_train3roles_testnewrole_probestatistics(num_persons_per_category):
    NUM_DIMS = 50
    NUM_TRAIN_EXAMPLES = 24000
    NUM_TEST_EXAMPLES = 120
    NUM_UNSEEN_TEST_EXAMPLES = 120
    NUM_UNSEEN_FILLERS = 100
    SAVE_PATH = os.path.join("/", "home", "cc27", "Thesis", "generalized_schema_learning", "data", "generate_train3roles_testnewrole_withunseentestfillers_%dpersonspercategory_%dtrain_%dtest" % (num_persons_per_category, NUM_TRAIN_EXAMPLES, NUM_TEST_EXAMPLES))

    STORY_FRAME = "begin subject sit subject friend announce emcee perform poet consume dessert drink goodbye".split(" ")
    QUESTIONS = ["QEmcee", "QFriend", "QPoet", "QSubject", "QDessert", "QDrink"]
    PADDING_WORD = "zzz"

    ROLES = ["emcee", "friend", "poet", "subject", "dessert", "drink"]
    NUM_PERSON_FILLERS = 4
    filler_indices = []
    for role in ROLES:
        filler_indices += list(np.where(np.array(STORY_FRAME) == role)[0])

    num_questions = len(QUESTIONS)
    person_fillers = [str(i) for i in range(num_persons_per_category * NUM_PERSON_FILLERS)]
    person_fillers_unseenintraining = [str(-1 * i) for i in range(1, NUM_UNSEEN_FILLERS + 1)]

    with open(os.path.join(SAVE_PATH, "wordslist.p"), "rb") as f:
        wordslist = pickle.load(f)

    # Determine fillers corresponding to the four person roles.
    person_fillers_indices = [wordslist.index(filler) for filler in person_fillers]
    person_fillers_unseenintraining_indices = [wordslist.index(filler) for filler in person_fillers_unseenintraining]
    person_fillers_bytrainmissingrole = {}
    for i in range(NUM_PERSON_FILLERS):
        person_fillers_bytrainmissingrole[ROLES[i]] = person_fillers_indices[i * num_persons_per_category : (i + 1) * num_persons_per_category]

    person_fillers_bytrainrole = {}
    for i in range(NUM_PERSON_FILLERS):
        role = ROLES[i]
        person_fillers_bytrainrole[role] = np.array(list(set(person_fillers_indices) - set(person_fillers_bytrainmissingrole[role])))

    person_fillers_indices = [wordslist.index(filler) for filler in person_fillers]
    person_fillers_unseenintraining_indices = [wordslist.index(filler) for filler in person_fillers_unseenintraining]
    person_fillers_bytrainmissingrole = {}
    for i in range(NUM_PERSON_FILLERS):
        person_fillers_bytrainmissingrole[ROLES[i]] = person_fillers_indices[i * num_persons_per_category : (i + 1) * num_persons_per_category]

    person_fillers_bytrainrole = {}
    for i in range(NUM_PERSON_FILLERS):
        role = ROLES[i]
        person_fillers_bytrainrole[role] = np.array(list(set(person_fillers_indices) - set(person_fillers_bytrainmissingrole[role])))

    # Get indices of fillers to replace.
    story_frame_matrix = np.expand_dims(np.expand_dims(np.array([wordslist.index(word) for word in STORY_FRAME]), axis=1), axis=0)
    PERSON_ROLES = ["emcee", "friend", "poet", "subject"]
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

    STORY_FRAME_PAD_UNTIL_SUBJECT = "zzz zzz zzz zzz friend announce emcee perform poet consume dessert drink goodbye".split(" ")
    STORY_FRAME_PAD_UNTIL_FRIEND = "zzz zzz zzz zzz zzz announce emcee perform poet consume dessert drink goodbye".split(" ")

    story_frame_manipulated_list = [STORY_FRAME_REPLACE_SUBJECT, STORY_FRAME_REPLACE_FRIEND, STORY_FRAME_REPLACE_EMCEE, STORY_FRAME_REPLACE_POET, STORY_FRAME_PAD_UNTIL_SUBJECT, STORY_FRAME_PAD_UNTIL_FRIEND]
    savename_list = ["replacesubject", "replacefriend", "replaceemcee", "replacepoet", "paduntilsubject", "paduntilfriend"]
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
                    filler = np.random.choice(person_fillers_bytrainmissingrole[role])
                    story[0, person_storyindices[role], 0] = filler
                answer = [story.squeeze()[question_storyindices[question]]]
                story = np.concatenate((story, padding, np.reshape(question, (1, 1, 1))), axis=1)
                split_test_X = np.concatenate((split_test_X, story), axis=0)
                split_test_y = np.concatenate((split_test_y, np.reshape(np.array(answer), (1, 1))), axis=0)
            print(wordslist[question], np.unique(split_test_y))
            with open(os.path.join(SAVE_PATH, "test_%s_%s.p" % (wordslist[question], savename)), "wb") as f:
                pickle.dump([split_test_X, split_test_y], f)

def generate_train3roles_testnewrole_withunseentestfillers_shuffledtestset(story_frame, shuffled_num, num_persons_per_category):
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
        with open(os.path.join(SAVE_PATH, "test_%s_shuffled%d_sameroles.p" % (wordslist[question], shuffled_num)), "wb") as f:
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
        with open(os.path.join(SAVE_PATH, "test_%s_shuffled%d_differentroles.p" % (wordslist[question], shuffled_num)), "wb") as f:
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
        with open(os.path.join(SAVE_PATH, "test_%s_shuffled%d_unseen.p" % (wordslist[question], shuffled_num)), "wb") as f:
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
            splitX, splity = np.load("test_%s_shuffled1_%s.p" % (key, description))
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

    print("TESTING SHUFFLED2 TEST SETS")
    for key in keys:
        for description in ["differentroles", "sameroles", "unseen"]:
            print(key, description)
            splitX, splity = np.load("test_%s_shuffled2_%s.p" % (key, description))
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
    #shuffled_1 = "consume dessert drink goodbye begin subject sit subject friend announce emcee perform poet".split(" ")
    #shuffled_2 = "begin subject sit subject friend perform poet announce emcee consume dessert drink goodbye".split(" ")
    #generate_train3roles_testnewrole_withunseentestfillers_shuffledtestset(shuffled_1, 1, 10)
    #generate_train3roles_testnewrole_withunseentestfillers_shuffledtestset(shuffled_2, 2, 10)
    generate_train3roles_testnewrole_probestatistics(1000)
