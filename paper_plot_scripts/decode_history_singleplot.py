"""
NOTE: RIDGE AND PRO ADAPTED FROM KIRAN'S SHERLOCK CODE
NOTE: Hard coded for specific sequence generated for analysis in variablefiller AllQs experiment.
"""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
import random
import sys

sys.path.append("../")
from numpy.linalg import svd
from sklearn.metrics.pairwise import cosine_similarity

TITLE_FONTSIZE = 20 
X_FONTSIZE = 14
AXIS_FONTSIZE = 16

filler_indices = {"Dessert": 21, 'Drink': 4, 'Emcee': 11, 'Friend': 9, 'Poet': 12, 'Subject': 1}  # Indices of fillers in the decoding story.

input_sequence = ['Begin', 'Subject', 'Order', 'Subject', 'Drink', 'Expensive', 'Subject', 'Sit', 'Subject', 'Friend', 'Intro', 'Emcee', 'Poet', 'Poetry', 'Poet', 'Decline', 'Subject', 'Goodbye', 'Subject', 'Friend', 'Order', 'Dessert', 'End', 'Subject', 'zzz', '?', 'QSubject']  # Sequence of words in the input story.

color_dict = {"Dessert": "#1f77b4", "Drink": "#ff7f0e", "Emcee": "#2ca02c", "Friend": "#d62728", "Poet": "#9467bd", "Subject": "#8c564b"}  # Color of line for each filler.
linestyle_dict = {"Dessert": "solid", "Drink": "dotted", "Emcee": "dashed", "Friend": "solid", "Poet": "dashdot", "Subject": "dashed"}  # Color of line for each filler.
fillers = ["Dessert", "Drink", "Emcee", "Friend", "Poet", "Subject"]  # Queried fillers.


def get_input_index(decoded_filler):
    """Returns index of desired filler in the story."""
    word_num = filler_indices[decoded_filler]
    return word_num


def get_one_data(timestep, data_option, train_indices, test_indices, decoded_filler, input_vectors, output_vectors, controller_histories_hiddenstate, memory_histories):
    """Returns network state data at a specified timestep.

    Args:
        timestep: Desired timestep of network state.
        data_option: Whether to obtain data from hidden state (`controller`) or enhanced memory (`memory`).
        train_indices: Samples used for training decoding map.
        test_indices: Samples used for testing decoding map.
        decoded_filler: Filler to decode.
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
        memory_size = memory_histories[0, :, :, timestep].size
        data_size = memory_size
    else:
        raise Exception("UNSUPPORTED DATA OPTION. Must be \"controller\" or \"memory\"")
    
    train_size = len(train_indices)
    X_train = np.zeros([train_size, data_size])
    Y_train = np.zeros([train_size, prediction_size])
    for i in range(train_size):
        train_index = train_indices[i]
        input_vector_index = get_input_index(decoded_filler)
        Y_train[i] = input_vectors[train_index][input_vector_index]
        try:
            if data_option == "controller":
                X_train[i] = controller_histories_hiddenstate[train_index, :, timestep].flatten()
            elif data_option == "memory":
                X_train[i] = memory_histories[train_index, :, :, timestep].flatten()
        except:
            import pdb; pdb.set_trace()

    test_size = len(test_indices)
    X_test = np.zeros([test_size, data_size])
    Y_test = np.zeros([test_size, prediction_size])
    for i in range(test_size):
        test_index = test_indices[i]
        input_vector_index = get_input_index(decoded_filler)
        Y_test[i] = input_vectors[test_index][input_vector_index]
        if data_option == "controller":
            X_test[i] = controller_histories_hiddenstate[test_index, :, timestep].flatten()
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
        UTy = np.ravel(UTy)  # NECESSARY TO HANDLE WEIRD MATRIX TYPES
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

def get_scores_byword(getdata_function, option, num_timesteps, train_indices, test_indices, decoded_filler, batch_embeddings, test_info, input_vectors, output_vectors, controller_histories_hiddenstate, memory_histories):
    '''Get the prediction score at each timestep.'''
    num_predictions = len(test_indices)
    allscores = np.zeros((len(input_sequence), num_predictions))
    W_ridges = []

    for i in range(num_predictions):
        for timestep in range(num_timesteps):
            X_train, Y_train, X_test, Y_test = getdata_function(timestep, option, train_indices, test_indices, decoded_filler, input_vectors, output_vectors, controller_histories_hiddenstate, memory_histories)
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
    return allscores, W_ridge

def run_decoding(num_epochs, network, data_dir, save_dir, trial_num, train_fraction=0.8):
    chance_rate = 0.5
    title_networks = {"LSTM-LN":"LSTM", "RNN-LN-FW":"Fast Weights", "RNN-LN":"RNN", "DNC":"DNC"}
    title_network = title_networks[network]
    test_filename = "test_analyze.npz"

    # Get saved states and embeddings.
    output_vectors = np.load(os.path.join(data_dir, 'output_vectors_%s_%depochs_trial%d_%s' % (network, num_epochs, trial_num, test_filename)))['arr_0']
    input_vectors = np.load(os.path.join(data_dir, 'input_vectors_%s_%depochs_trial%d_%s' % (network, num_epochs, trial_num, test_filename)))['arr_0']
    controller_histories_hiddenstate = np.load(os.path.join(data_dir, 'hidden_histories_%s_%depochs_trial%d_%s' % (network, num_epochs, trial_num, test_filename)))['arr_0']
    memory_histories = None
    if network in ["RNN-LN-FW", "DNC"]:
        memory_histories = np.load(os.path.join(data_dir, 'memory_histories_%s_%depochs_trial%d_%s' % (network, num_epochs, trial_num, test_filename)))['arr_0']
    batch_embeddings = np.load(os.path.join(data_dir, 'batch_embeddings_%s_%depochs_trial%d_%s' % (network, num_epochs, trial_num, test_filename)))['arr_0']
    batch_embeddings = np.unique(batch_embeddings, axis=0)  # Some batch embeddings are repeated because frame words are repeated in each example.
    with open(os.path.join(data_dir, ('test_analysis_results_%s_%depochs_trial%d_%s' % (network, num_epochs, trial_num, test_filename))).replace("npz","p"), 'rb') as f:
        test_info = pickle.load(f)
    num_examples, num_timesteps, _ = input_vectors.shape
    all_indices = range(num_examples)
    train_indices = all_indices[:int(train_fraction * num_examples)]
    test_indices = all_indices[int(train_fraction * num_examples):]
    plot_memory_and_controller = network in ["DNC", "RNN-LN-FW"]  # Only plot the memory states for the networks with memory states.
    input_sequence_length = len(input_sequence)
    x_indices = range(input_sequence_length)

    # Plot scores.
    for decoded_filler in fillers:
        decoded_filler_color = color_dict[decoded_filler]
        decoded_filler_linestyle = linestyle_dict[decoded_filler]
        scores_controller, W_ridges_controller = get_scores_byword(get_one_data, "controller", num_timesteps, train_indices, test_indices, decoded_filler, batch_embeddings, test_info, input_vectors, output_vectors, controller_histories_hiddenstate, memory_histories)
        if plot_memory_and_controller:
            scores_memory, W_ridges_memory = get_scores_byword(get_one_data, "memory", num_timesteps, train_indices, test_indices, decoded_filler, batch_embeddings, test_info, input_vectors, output_vectors, controller_histories_hiddenstate, memory_histories)
            plt.subplot(211)
            plt.errorbar(x_indices, np.average(scores_memory, axis=1), label=decoded_filler, color=decoded_filler_color, linestyle=decoded_filler_linestyle)
            if "DNC" in network:
                plt.ylabel("External Memory", fontsize=AXIS_FONTSIZE)
            elif "RNN-LN-FW" in network:
                plt.ylabel("FW Matrix", fontsize=AXIS_FONTSIZE)
            plt.grid(color='grey', linestyle='-', linewidth=1, alpha=0.5)
            plt.subplot(212)
            plt.errorbar(x_indices, np.average(scores_controller, axis=1), label=decoded_filler, color=decoded_filler_color, linestyle=decoded_filler_linestyle)
        else:
            plt.errorbar(x_indices, np.average(scores_controller, axis=1), label=decoded_filler, color=decoded_filler_color, linestyle=decoded_filler_linestyle)
            plt.grid(color='grey', linestyle='-', linewidth=1, alpha=0.5)
        for i in x_indices:
            if input_sequence[i] == decoded_filler:
                if plot_memory_and_controller:
                    plt.subplot(211)
                    plt.scatter([i], np.average(scores_memory, axis=1)[i], color=decoded_filler_color)
                    plt.subplot(212)
                    plt.scatter([i], np.average(scores_controller, axis=1)[i], color=decoded_filler_color)
                else:
                    plt.scatter([i], np.average(scores_controller, axis=1)[i], color=decoded_filler_color)
    if plot_memory_and_controller:
        plt.subplot(211)
        plt.title(title_network, fontsize=TITLE_FONTSIZE)
        plt.axhline(y=chance_rate, label="Chance Rate", color='k', linestyle='--')
        plt.xticks(x_indices, [""] * len(input_sequence))
        plt.subplot(212)
        plt.axhline(y=chance_rate, label="Chance Rate", color='k', linestyle='--')
    else:
        plt.axhline(y=chance_rate, label="Chance Rate", color='k', linestyle='--')
        plt.title(title_network, fontsize=TITLE_FONTSIZE)
    plt.xticks(x_indices, input_sequence, fontsize=X_FONTSIZE, rotation=90)
    # Set label colors.
    for xtick, xticklabel in zip(plt.gca().get_xticklabels(), input_sequence):
        if xticklabel in color_dict.keys():
            xtick.set_color(color_dict[xticklabel])
    plt.grid(color='grey', linestyle='-', linewidth=1, alpha=0.5)
    plt.ylim([0,1])
    plt.ylabel("Hidden State", fontsize=AXIS_FONTSIZE)
    plt.savefig(os.path.join(save_dir, ("experiment_variable_filler_trial%d_%depochs_%s_ranking_trial%d" % (trial_num, num_epochs, network, trial_num))), bbox_inches='tight')
    plt.close()


def run_decoding_trial_averaged(num_epochs, network, data_dir, save_dir, trial_nums, train_fraction=0.8):
    chance_rate = 0.5
    title_networks = {"LSTM-LN":"LSTM", "RNN-LN-FW":"Fast Weights", "RNN-LN":"RNN", "DNC":"DNC"}
    title_network = title_networks[network]
    test_filename = "test_analyze.npz"

    # Get saved states and embeddings.
    scores_controller_dict = {}
    scores_memory_dict = {}
    for trial_num in trial_nums:
        output_vectors = np.load(os.path.join(data_dir, 'output_vectors_%s_%depochs_trial%d_%s' % (network, num_epochs, trial_num, test_filename)))['arr_0']
        input_vectors = np.load(os.path.join(data_dir, 'input_vectors_%s_%depochs_trial%d_%s' % (network, num_epochs, trial_num, test_filename)))['arr_0']
        controller_histories_hiddenstate = np.load(os.path.join(data_dir, 'hidden_histories_%s_%depochs_trial%d_%s' % (network, num_epochs, trial_num, test_filename)))['arr_0']
        memory_histories = None
        if network in ["RNN-LN-FW", "DNC"]:
            memory_histories = np.load(os.path.join(data_dir, 'memory_histories_%s_%depochs_trial%d_%s' % (network, num_epochs, trial_num, test_filename)))['arr_0']
        batch_embeddings = np.load(os.path.join(data_dir, 'batch_embeddings_%s_%depochs_trial%d_%s' % (network, num_epochs, trial_num, test_filename)))['arr_0']
        batch_embeddings = np.unique(batch_embeddings, axis=0)  # Some batch embeddings are repeated because frame words are repeated in each example.
        with open(os.path.join(data_dir, ('test_analysis_results_%s_%depochs_trial%d_%s' % (network, num_epochs, trial_num, test_filename))).replace("npz","p"), 'rb') as f:
            test_info = pickle.load(f)
        num_examples, num_timesteps, _ = input_vectors.shape
        all_indices = range(num_examples)
        train_indices = all_indices[:int(train_fraction * num_examples)]
        test_indices = all_indices[int(train_fraction * num_examples):]
        plot_memory_and_controller = network in ["DNC", "RNN-LN-FW"]  # Only plot the memory states for the networks with memory states.
        input_sequence_length = len(input_sequence)
        x_indices = range(input_sequence_length)
        for decoded_filler in fillers:
            scores_controller, W_ridges_controller = get_scores_byword(get_one_data, "controller", num_timesteps, train_indices, test_indices, decoded_filler, batch_embeddings, test_info, input_vectors, output_vectors, controller_histories_hiddenstate, memory_histories)
            if decoded_filler not in scores_controller_dict:
                scores_controller_dict[decoded_filler] = scores_controller
            else:
                scores_controller_dict[decoded_filler] = np.concatenate((scores_controller_dict[decoded_filler], scores_controller), axis=1)
            if plot_memory_and_controller:
                scores_memory, W_ridges_memory = get_scores_byword(get_one_data, "memory", num_timesteps, train_indices, test_indices, decoded_filler, batch_embeddings, test_info, input_vectors, output_vectors, controller_histories_hiddenstate, memory_histories)
                if decoded_filler not in scores_memory_dict:
                    scores_memory_dict[decoded_filler] = scores_memory
                else:
                    scores_memory_dict[decoded_filler] = np.concatenate((scores_memory_dict[decoded_filler], scores_memory), axis=1)

    # Plot scores.
    for decoded_filler in fillers:
        decoded_filler_color = color_dict[decoded_filler]
        decoded_filler_linestyle = linestyle_dict[decoded_filler]
        if plot_memory_and_controller:
            plt.subplot(211)
            plt.errorbar(x_indices, np.average(scores_memory_dict[decoded_filler], axis=1), label=decoded_filler, color=decoded_filler_color, linestyle=decoded_filler_linestyle)
            if "DNC" in network:
                plt.ylabel("External Memory", fontsize=AXIS_FONTSIZE)
            elif "RNN-LN-FW" in network:
                plt.ylabel("FW Matrix", fontsize=AXIS_FONTSIZE)
            plt.grid(color='grey', linestyle='-', linewidth=1, alpha=0.5)
            plt.subplot(212)
            plt.errorbar(x_indices, np.average(scores_controller_dict[decoded_filler], axis=1), label=decoded_filler, color=decoded_filler_color, linestyle=decoded_filler_linestyle)
        else:
            plt.errorbar(x_indices, np.average(scores_controller_dict[decoded_filler], axis=1), label=decoded_filler, color=decoded_filler_color, linestyle=decoded_filler_linestyle)
            plt.grid(color='grey', linestyle='-', linewidth=1, alpha=0.5)
        for i in x_indices:
            if input_sequence[i] == decoded_filler:
                if plot_memory_and_controller:
                    plt.subplot(211)
                    plt.scatter([i], np.average(scores_memory_dict[decoded_filler], axis=1)[i], color=decoded_filler_color)
                    plt.subplot(212)
                    plt.scatter([i], np.average(scores_controller_dict[decoded_filler], axis=1)[i], color=decoded_filler_color)
                else:
                    plt.scatter([i], np.average(scores_controller_dict[decoded_filler], axis=1)[i], color=decoded_filler_color)
    if plot_memory_and_controller:
        plt.subplot(211)
        plt.title(title_network, fontsize=TITLE_FONTSIZE)
        plt.axhline(y=chance_rate, label="Chance Rate", color='k', linestyle='--')
        plt.xticks(x_indices, [""] * len(input_sequence))
        plt.subplot(212)
        plt.axhline(y=chance_rate, label="Chance Rate", color='k', linestyle='--')
    else:
        plt.axhline(y=chance_rate, label="Chance Rate", color='k', linestyle='--')
        plt.title(title_network, fontsize=TITLE_FONTSIZE)
    plt.xticks(x_indices, input_sequence, fontsize=X_FONTSIZE, rotation=90)
    # Set label colors.
    for xtick, xticklabel in zip(plt.gca().get_xticklabels(), input_sequence):
        if xticklabel in color_dict.keys():
            xtick.set_color(color_dict[xticklabel])
    plt.grid(color='grey', linestyle='-', linewidth=1, alpha=0.5)
    plt.ylim([0,1])
    plt.ylabel("Hidden State", fontsize=AXIS_FONTSIZE)
    plt.savefig(os.path.join(save_dir, ("experiment_variable_filler_%depochs_%s_ranking_trial_average" % (num_epochs, network))), bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # Experiment: Subject, AllQs, etc.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epochs', type=int, default='30000', help='Number of epochs for which the network was trained.')
    parser.add_argument('--network', type=str, help='Name of network to perform decoding on.')
    parser.add_argument('--trial-nums', type=int, help='Trial number to use for analysis.', default=10, nargs='+')
    parser.add_argument('--data-dir', type=str, default=os.path.join('variable_filler', 'analysis'), help='Directory containing saved network states.')
    parser.add_argument('--save-dir', type=str, default='decoding_results', help='Directory in which to save decoding results.')
    args = parser.parse_args()
    run_decoding_trial_averaged(num_epochs=args.num_epochs,
            network=args.network,
            data_dir=args.data_dir,
            save_dir=args.save_dir,
            trial_nums=args.trial_nums)
    for trial_num in args.trial_nums:
        run_decoding(num_epochs=args.num_epochs,
                network=args.network,
                data_dir=args.data_dir,
                save_dir=args.save_dir,
                trial_num=trial_num)
