"""
NOTE: RIDGE AND PRO ADAPTED FROM KIRAN'S SHERLOCK CODE
NOTE: Hard coded for specific sequence generated for analysis in variablefiller AllQs experiment.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import pickle
import random
import sys

sys.path.append("../")
from directories import base_dir
from numpy.linalg import svd
from sklearn.metrics.pairwise import cosine_similarity

TITLE_FONTSIZE = 12
X_FONTSIZE = 8

filler_indices = {
    "QSubjectQFriend":{"Dessert":23, 'Drink':4, 'Emcee':11, 'Friend':9, 'Poet':12, 'Subject':1},
    "QSubject":{"Dessert":23, 'Drink':4, 'Emcee':11, 'Friend':9, 'Poet':12, 'Subject':1},
    "AllQs":{"Dessert":23, 'Drink':4, 'Emcee':11, 'Friend':9, 'Poet':12, 'Subject':1}
}

input_sequences = {
    "QSubjectQFriend": ['BEGIN', 'Subject', 'Order_drink', 'Subject', 'Drink', 'Too_expensive', 'Subject', 'Sit_down', 'Subject', 'Friend', 'Emcee_intro', 'Emcee', 'Poet', 'Poet_performs', 'Poet', 'Subject_performs', 'Subject', 'Friend', 'Say_goodbye', 'Subject', 'Friend', 'Order_dessert', 'Subject', 'Dessert', 'END', 'Subject', 'zzz', '?', 'QSubject'],
    "QSubject": ['BEGIN', 'Subject', 'Order_drink', 'Subject', 'Drink', 'Too_expensive', 'Subject', 'Sit_down', 'Subject', 'Friend', 'Emcee_intro', 'Emcee', 'Poet', 'Poet_performs', 'Poet', 'Subject_performs', 'Subject', 'Friend', 'Say_goodbye', 'Subject', 'Friend', 'Order_dessert', 'Subject', 'Dessert', 'END', 'Subject', 'zzz', '?', 'QSubject'],
    "AllQs": ['BEGIN', 'Subject', 'Order_drink', 'Subject', 'Drink', 'Too_expensive', 'Subject', 'Sit_down', 'Subject', 'Friend', 'Emcee_intro', 'Emcee', 'Poet', 'Poet_performs', 'Poet', 'Subject_performs', 'Subject', 'Friend', 'Say_goodbye', 'Subject', 'Friend', 'Order_dessert', 'Subject', 'Dessert', 'END', 'Subject', 'zzz', '?', 'QEmcee']
}
def get_input_index(experiment_name, Y_type):
    """Returns index of desired filler in the story."""
    word_num = filler_indices[experiment_name][Y_type]
    return word_num

def get_one_data(timestep, data_option, train_indices, test_indices, Y_type, input_vectors, output_vectors, controller_histories_hiddenstate, memory_histories, experiment_name):
    """Returns network state data at a specified timestep.

    Args:
        timestep: Desired timestep of network state.
        data_option: Whether to obtain data from hidden state (`controller`) or enhanced memory (`memory`).
        train_indices: Samples used for training decoding map.
        test_indices: Samples used for testing decoding map.
        Y_type: Filler to decode.
        input_vectors: Input sequences.
        output_vectors: Network's outputs.
        controller_histories_hiddenstate: Saved controller histories.
        memory_histories: Saved enhanced memory component histories.

    Returns:
        X_train: Network states (to train mapping).
        y_train: Filler to decode (to train mapping).
        X_test: Network states (to test mapping).
        y_test: Filler to decode (to test mapping).
    """
    prediction_size = output_vectors.shape[1]
    if data_option == "controller":
        hiddenstate_size = controller_histories_hiddenstate.shape[1]
        data_size = hiddenstate_size
    elif data_option == "memory":
        memory_size = memory_histories[0,:,:,timestep].size
        data_size = memory_size
    else:
        raise ArgumentError("UNSUPPORTED DATA OPTION. Must be \"controller\" or \"memory\"")

    train_size = len(train_indices)
    X_train = np.zeros([train_size, data_size])
    Y_train = np.zeros([train_size, prediction_size])
    for i in range(train_size):
        train_index = train_indices[i]
        input_vector_index = get_input_index(experiment_name, Y_type)
        Y_train[i] = input_vectors[train_index][input_vector_index]
        if data_option == "controller":
            X_train[i] = controller_histories_hiddenstate[train_index,:,timestep].flatten()
        elif data_option == "memory":
            X_train[i] = memory_histories[train_index,:,:,timestep].flatten()

    test_size = len(test_indices)
    X_test = np.zeros([test_size, data_size])
    Y_test = np.zeros([test_size, prediction_size])
    for i in range(test_size):
        test_index = test_indices[i]
        input_vector_index = get_input_index(experiment_name, Y_type)
        Y_test[i] = input_vectors[test_index][input_vector_index]
        if data_option == "controller":
            X_test[i] = controller_histories_hiddenstate[test_index,:,timestep].flatten()
        elif data_option == "memory":
            X_test[i] = memory_histories[test_index,:,:,timestep].flatten()
    return X_train, Y_train, X_test, Y_test

# Y = XW.
def ridge_fit(ridge_param, X_train, Y_train):
    """Learns a mapping using ridge regression.

    Args:
        ridge_param: Parameter for ridge regression.
        X_train: [n_samples x n_features] matrix of network states.
        Y_train: [n_samples x output_dim] matrix of fillers.
    """
    num_features = np.shape(X_train)[1]
    U, s, VT = svd(X_train, full_matrices=False)
    V = VT.T
    scaled_d = np.zeros(np.shape(s))
    for i in range(0, len(scaled_d)):
        scaled_d[i] = s[i]/(s[i]*s[i] + ridge_param)
    # Y: n_samples x n_targets
    n_targets = np.shape(Y_train)[1]
    W = np.zeros((num_features, n_targets))
    for k in range(0, n_targets):
        y = Y_train[:, k]
        UTy = np.dot(U.T, y)
        UTy = np.ravel(UTy) # NECESSARY TO HANDLE WEIRD MATRIX TYPES
        dUTy = scaled_d*UTy
        w_k = np.dot(V, dUTy)
        W[:, k] = w_k
    # Return n_features x n_targets
    return W

def score_prediction(prediction, batch_embedding, Yi):
    """Computes the ranking score of a prediction (change rate 50%).

    Args:
        prediction: Predicted vector.
        batch_embedding: Matrix of all vectors in corpus.
        Yi: Correct vector.
    """
    prediction_similarities = cosine_similarity(batch_embedding, prediction)
    actual_index = np.argmax(cosine_similarity(batch_embedding, Yi))
    actual_index_similarity = prediction_similarities[actual_index]
    corpus_size = batch_embedding.shape[0]
    ranking = np.count_nonzero(prediction_similarities < actual_index_similarity) * 1.0 / (corpus_size)
    return ranking

def get_scores_byword(getdata_function, option, num_timesteps, train_indices, test_indices, Y_type, batch_embeddings, test_info, input_vectors, output_vectors, controller_histories_hiddenstate, memory_histories, experiment):
    scores_index = ["begin", "begin_SUBJECT", #0,1
                    "orderdr", "orderdr_SUBJECT", "orderdr_DRINK", #2,3,4
                    "expensive", "expensive_SUBJECT", #5,6
                    "sit", "sit_SUBJECT", "sit_FRIEND", #7,8,9
                    "emceeintro", "emceeintro_EMCEE", "emceeintro_POET", #10,11,12
                    "poetp", "poetp_POET", #13,14
                    "subjectp", "subjectp_SUBJECT", "subjectp_FRIEND", #15,16,17
                    "bye", "bye_SUBJECT", "bye_FRIEND", #18,19,20
                    "orderde", "orderde_SUBJECT", "orderde_DESSERT", #21,22,23
                    "end", "end_SUBJECT", #24,25
                    "zzz", #26
                    "?", #27
                    "Query", #28
                    ]
    num_predictions = len(test_indices)
    allscores = np.zeros((len(scores_index), num_predictions))
    W_ridges = []

    for i in range(num_predictions):
        for timestep in range(num_timesteps):
            X_train, Y_train, X_test, Y_test = getdata_function(timestep, option, train_indices, test_indices, Y_type, input_vectors, output_vectors, controller_histories_hiddenstate, memory_histories, experiment_name)
            # Fit.
            ridge_param = 1.0
            if i == 0:
                W_ridge = ridge_fit(ridge_param, X_train, Y_train)
                W_ridges.append(W_ridge)
            else:
                W_ridge = W_ridges[timestep]

            # Predict and Score.
            X = X_test
            Y = Y_test
            Xi = np.expand_dims(X[i], axis=0)
            Yi = np.expand_dims(Y[i], axis=0)
            prediction_ridge = np.dot(Xi, W_ridge)
            ranking_score = score_prediction(prediction_ridge, batch_embeddings, Yi)

            # Fill in scores.
            allscores[timestep, i] = ranking_score
    return allscores, W_ridges

def run_trial(experiment_name, previous_trained_epochs, model_name, historypath, savepath, input_sequence, trial_num):
    chance_rate = 0.5
    title_model_names = {"LSTM-LN":"LSTM", "RNN-LN-FW":"Fast Weights", "RNN-LN":"RNN", "NTM2":"Reduced NTM"}
    title_model_name = title_model_names[model_name]
    test_filename = "test_analyze.npz"
    with open(os.path.join(historypath, 'output_vectors_%s_%depochs_trial%d_%s' % (model_name, previous_trained_epochs, trial_num, test_filename))) as f:
        output_vectors = np.load(f)['arr_0']
    with open(os.path.join(historypath, 'input_vectors_%s_%depochs_trial%d_%s' % (model_name, previous_trained_epochs, trial_num, test_filename))) as f:
        input_vectors = np.load(f)['arr_0']
    with open(os.path.join(historypath, 'hidden_histories_%s_%depochs_trial%d_%s' % (model_name, previous_trained_epochs, trial_num, test_filename))) as f:
        controller_histories_hiddenstate = np.load(f)['arr_0']
    memory_histories = None
    if model_name in ["RNN-LN-FW", "NTM2"]:
        with open(os.path.join(historypath, 'memory_histories_%s_%depochs_trial%d_%s' % (model_name, previous_trained_epochs, trial_num, test_filename))) as f:
            memory_histories = np.load(f)['arr_0']
    with open(os.path.join(historypath, 'batch_embeddings_%s_%depochs_trial%d_%s' % (model_name, previous_trained_epochs, trial_num, test_filename))) as f:
        batch_embeddings = np.load(f)['arr_0']
        batch_embeddings = np.unique(batch_embeddings, axis=0)
    with open(os.path.join(historypath, ('test_analysis_results_%s_%depochs_trial%d_%s' % (model_name, previous_trained_epochs, trial_num, test_filename))).replace("npz","p")) as f:
        test_info = pickle.load(f)
    num_timesteps = 29
    num_examples = output_vectors.shape[0]
    train_fraction = 0.8
    all_indices = range(num_examples)
    random.shuffle(all_indices)
    train_indices = all_indices[:int(train_fraction * num_examples)]
    test_indices = all_indices[int(train_fraction * num_examples):]
    plot_memory_and_controller = model_name in ["NTM2", "RNN-LN-FW"]
    input_sequence_length = len(input_sequence)
    xrange = range(input_sequence_length)
    #color_dict = {"Dessert":"#1f77b4", "Drink":"#ff7f0e", "Emcee":"#2ca02c", "Friend":"#d62728", "Poet":"#9467bd", "Subject":"#8c564b"}
    color_dict = {"Dessert":"#006BA4", "Drink":"#FF800E", "Emcee":"#ABABAB", "Friend":"#595959", "Poet":"#5F9ED1", "Subject":"#C85200"}
    line_style_dict = {"Dessert":"solid", "Drink":"dotted", "Emcee":"dashed", "Friend":"solid", "Poet":"dotted", "Subject":"dashed"}
    line_width_dict = {"Dessert":0.8, "Drink":0.8, "Emcee":0.8, "Friend":1.5, "Poet":1.5, "Subject":1.5}
    line_marker_dict = {"Dessert":".", "Drink":".", "Emcee":".", "Friend":".", "Poet":".", "Subject":"."}
    # Plot scores.
    for Y_type in ["Dessert", "Drink", "Emcee", "Friend", "Poet", "Subject"]:
        Y_type_color = color_dict[Y_type]
        Y_type_line_style = line_style_dict[Y_type]
        Y_type_line_width = line_width_dict[Y_type]
        Y_line_marker = line_marker_dict[Y_type]
        scores_controller, W_ridges_controller = get_scores_byword(get_one_data, "controller", num_timesteps, train_indices, test_indices, Y_type, batch_embeddings, test_info, input_vectors, output_vectors, controller_histories_hiddenstate, memory_histories, experiment_name)
        print(os.path.join(savepath, "experiment%s_trial%d_%depochs_%s_ranking_%s_controller_W_ridges" % (experiment_name, trial_num, previous_trained_epochs, Y_type, model_name)))
        with open(os.path.join(savepath, "experiment%s_trial%d_%depochs_%s_ranking_%s_controller_W_ridges" % (experiment_name, trial_num, previous_trained_epochs, Y_type, model_name)), "wb") as f:
            pickle.dump(W_ridges_controller, f)
        if plot_memory_and_controller:
            scores_memory, W_ridges_memory = get_scores_byword(get_one_data, "memory", num_timesteps, train_indices, test_indices, Y_type, batch_embeddings, test_info, input_vectors, output_vectors, controller_histories_hiddenstate, memory_histories, experiment_name)
            print(os.path.join(savepath, "experiment%s_trial%d_%depochs_%s_ranking_%s_memory_W_ridges" % (experiment_name, trial_num, previous_trained_epochs, Y_type, model_name)))
            with open(os.path.join(savepath, "experiment%s_trial%d_%depochs_%s_ranking_%s_memory_W_ridges" % (experiment_name, trial_num, previous_trained_epochs, Y_type, model_name)), "wb") as f:
                pickle.dump(W_ridges_memory, f)
            plt.subplot(211)
            plt.errorbar(xrange, np.average(scores_memory, axis=1), label=Y_type, color=Y_type_color, linestyle=Y_type_line_style, linewidth=Y_type_line_width, marker=Y_line_marker)
            if "NTM2" in model_name:
                plt.ylabel("External Memory")
            elif "RNN-LN-FW" in model_name:
                plt.ylabel("FW Matrix")
            plt.grid(color='grey', linestyle='-', linewidth=1, alpha=0.5)
            plt.subplot(212)
            plt.errorbar(xrange, np.average(scores_controller, axis=1), label=Y_type, color=Y_type_color, linestyle=Y_type_line_style, linewidth=Y_type_line_width, marker=Y_line_marker)
        else:
            plt.errorbar(xrange, np.average(scores_controller, axis=1), label=Y_type, color=Y_type_color, linestyle=Y_type_line_style, linewidth=Y_type_line_width, marker=Y_line_marker)
            plt.grid(color='grey', linestyle='-', linewidth=1, alpha=0.5)
        for i in xrange:
            if input_sequence[i] == Y_type:
                if plot_memory_and_controller:
                    plt.subplot(211)
                    plt.scatter([i], np.average(scores_memory, axis=1)[i], color=Y_type_color, linestyle=Y_type_line_style, linewidth=Y_type_line_width, marker=Y_line_marker)
                    # plt.errorbar(xrange, np.average(scores_memory, axis=1), yerr = np.std(scores_memory, axis=1))
                    plt.subplot(212)
                    plt.scatter([i], np.average(scores_controller, axis=1)[i], color=Y_type_color, linestyle=Y_type_line_style, linewidth=Y_type_line_width, marker=Y_line_marker)
                    # plt.errorbar(xrange, np.average(scores_controller, axis=1), yerr = np.std(scores_controller, axis=1))
                else:
                    plt.scatter([i], np.average(scores_controller, axis=1)[i], color=Y_type_color, linestyle=Y_type_line_style, linewidth=Y_type_line_width, marker=Y_line_marker)
                    # plt.errorbar(xrange, np.average(scores_controller, axis=1), yerr = np.std(scores_controller, axis=1))
    if plot_memory_and_controller:
        plt.subplot(211)
        plt.title("%s Trained for %d Epochs" % (title_model_name, previous_trained_epochs), fontsize=TITLE_FONTSIZE)
        plt.axhline(y=chance_rate, label="Chance Rate", color='k', linestyle='--')
        plt.xticks(xrange, [""] * len(input_sequence))
        plt.subplot(212)
        plt.axhline(y=chance_rate, label="Chance Rate", color='k', linestyle='--')
        # legend = plt.legend(bbox_to_anchor=(1, 1.5))
    else:
        plt.axhline(y=chance_rate, label="Chance Rate", color='k', linestyle='--')
        # legend = plt.legend(bbox_to_anchor=(1, 0.5))
        plt.title("%s Trained for %d Epochs" % (title_model_name, previous_trained_epochs), fontsize=TITLE_FONTSIZE)
    plt.xticks(xrange, input_sequence, fontsize=X_FONTSIZE, rotation=90)
    # Set label colors.
    for xtick, xticklabel in zip(plt.gca().get_xticklabels(), input_sequence):
        if xticklabel in color_dict.keys():
            xtick.set_color(color_dict[xticklabel])
    plt.grid(color='grey', linestyle='-', linewidth=1, alpha=0.5)
    plt.ylim([0,1])
    plt.ylabel("Hidden State")
    plt.xlabel("Input Word")
    #legend = plt.legend(ncol=2, bbox_to_anchor=(1, -1))
    plt.savefig(os.path.join(savepath, ("experiment%s_trial%d_%depochs_%s_ranking" % (experiment_name, trial_num, previous_trained_epochs, model_name))), bbox_inches='tight', dpi=1200)
    #plt.savefig(os.path.join(savepath, 'legend'))
    plt.close()

if __name__ == '__main__':
    # Experiment: Subject, AllQs, etc.
    # Y_type: Filler to decode (Subject, Friend, etc.)
    experiment_name = sys.argv[1]
    previous_trained_epochs = int(sys.argv[2])
    model_name = sys.argv[3]
    historypath = os.path.join(base_dir, "results", "variablefiller_gensymbolicstates_100000_1_testunseen_%s" % experiment_name, "variable_filler", "analysis")
    if not os.path.exists(historypath):
        os.makedirs(historypath)
    savepath = os.path.join(base_dir, "figures_20200121")
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    input_sequence = input_sequences[experiment_name]
    trial_num = 0
    run_trial(experiment_name, previous_trained_epochs, model_name, historypath, savepath, input_sequence, trial_num)
