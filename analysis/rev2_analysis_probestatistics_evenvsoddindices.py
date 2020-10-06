import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

PERCENTAGE_INDISTRIBUTION = 0
SUBCATEGORY_BAR_WIDTH = 1.0 / 15

def get_predictions(filename):
    with open(filename, "rb") as f:
        predictions = pickle.load(f)
    predictions_concatenated = predictions[0]
    for i in range(1, len(predictions)):
        predictions_concatenated = np.concatenate((predictions_concatenated, predictions[i]), axis=0)
    print("mean odd", np.mean(predictions_concatenated[:,1::2]))
    print("mean even", np.mean(predictions_concatenated[:,::2]))
    return predictions_concatenated

def get_predictions_and_filename(filenames, results_data_dir, network_name, trial_name):
    try:
        network_filename = [filename for filename in filenames if ("logits_{network_name}".format(network_name=network_name) in filename) and (trial_name in filename)][-1]
        with open(os.path.join(results_data_dir, network_filename), "rb") as f:
            predictions = pickle.load(f)
        predictions_concatenated = predictions[0]
        for i in range(1, len(predictions)):
            predictions_concatenated = np.concatenate((predictions_concatenated, predictions[i]), axis=0)
        print("mean odd", np.mean(predictions_concatenated[:,1::2]))
        print("mean even", np.mean(predictions_concatenated[:,::2]))
        return predictions_concatenated
    except Exception as e:
        print(results_data_dir, network_name)
        return np.zeros((1, 50))

for trial_name in ["trial0", "trial1", "trial2"]:
    for role in ["subject", "poet", "emcee", "friend"]:
        plt_inds_odd = []
        plt_inds_even = []
        plt_labels = []
        odd_values = []
        even_values = []
        bbox_extra_artists = []
        for i, percentage_indistribution in enumerate([0, 25, 50, 75, 100]):
            RESULTS_DATA_DIR = "/home/cc27/Thesis/generalized_schema_learning/results/probestatisticsretention_percentageindistribution{percentage_indistribution}_normalizefillerdistributionFalse/fixed_filler/probe_statistics_test_Q{role_capitalized}_replace{role}".format(percentage_indistribution=percentage_indistribution, role_capitalized=role.title(), role=role)

            filenames = os.listdir(RESULTS_DATA_DIR)

            # lstm_file = [filename for filename in filenames if ("logits_LSTM-LN" in filename) and (trial_name in filename)][-1]
            # rnn_file = [filename for filename in filenames if ("logits_RNN-LN" in filename) and (trial_name in filename)][-1]
            # fw_file = [filename for filename in filenames if ("logits_RNN-LN-FW" in filename) and (trial_name in filename)][-1]
            # ntm2_file = [filename for filename in filenames if ("logits_NTM2" in filename) and (trial_name in filename)][-1]
            #
            # lstm = get_predictions(os.path.join(RESULTS_DATA_DIR, lstm_file))
            # rnn = get_predictions(os.path.join(RESULTS_DATA_DIR, rnn_file))
            # fw = get_predictions(os.path.join(RESULTS_DATA_DIR, fw_file))
            # ntm2 = get_predictions(os.path.join(RESULTS_DATA_DIR, ntm2_file))
            lstm = get_predictions_and_filename(filenames, RESULTS_DATA_DIR, "LSTM-LN", trial_name)
            rnn = get_predictions_and_filename(filenames, RESULTS_DATA_DIR, "RNN-LN", trial_name)
            fw = get_predictions_and_filename(filenames, RESULTS_DATA_DIR, "RNN-LN-FW", trial_name)
            ntm2 = get_predictions_and_filename(filenames, RESULTS_DATA_DIR, "NTM2", trial_name)

            all_results = [rnn, lstm, fw, ntm2]
            plt_inds_odd += [i + SUBCATEGORY_BAR_WIDTH * 3 * j for j in range(4)]
            plt_inds_even += [i + SUBCATEGORY_BAR_WIDTH * 3 * j + SUBCATEGORY_BAR_WIDTH for j in range(4)]
            #plt_labels += ["rnn odd", "rnn even", "lstm odd", "lstm even", "fw odd", "fw even", "ntm odd", "ntm even"]
            plt_labels += ["rnn", "lstm", "fw", "ntm"]
            odd_values += [np.mean(results[:,1::2]) for results in all_results]
            even_values += [np.mean(results[:,::2]) for results in all_results]
            text = plt.text(i, -0.35, percentage_indistribution, fontsize=12)
            bbox_extra_artists.append(text)

        #plt.bar(plt_inds_odd, odd_values, width=SUBCATEGORY_BAR_WIDTH*2/3, label="Mean Odd Indices", color="#1f77b4")
        #plt.bar(plt_inds_even, even_values, width=SUBCATEGORY_BAR_WIDTH*2/3, label="Mean Even Indices", color="#ff7f0e")
        plt.bar(plt_inds_odd, np.array(even_values) - np.array(odd_values), width=SUBCATEGORY_BAR_WIDTH, color="#1f77b4")
        #plt.ylim([-0.4, 0.4])
        plt.ylim([-0.7, 0.7])
        #locs = plt_inds_odd + plt_inds_even
        locs = plt_inds_odd
        locs.sort()
        plt.xticks(locs, plt_labels, rotation=90, fontsize=6)
        #legend = plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        #plt.title("Even vs odd indices for query Q{role}".format(role=role))
        plt.title("Even minus odd indices for query Q{role}".format(role=role))
        #bbox_extra_artists.append(legend)
        #plt.savefig("/home/cc27/Thesis/generalized_schema_learning/figures/probedistributions/evenvsodd/{trial_name}_{role}".format(trial_name=trial_name, role=role), bbox_extra_artists=bbox_extra_artists, bbox_inches="tight")
        plt.savefig("/home/cc27/Thesis/generalized_schema_learning/figures/probedistributions/evenvsodd/diffs_{trial_name}_{role}".format(trial_name=trial_name, role=role), bbox_inches="tight")
        plt.close()
