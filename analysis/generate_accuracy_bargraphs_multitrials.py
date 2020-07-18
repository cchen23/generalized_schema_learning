"""Modules to generate custom plots.

Assumes results stored using train module in run_experiment.py.
"""
import argparse
import ast
import fnmatch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys
sys.path.append("../")
from directories import base_dir

TRAIN_INDEX = 0
TEST_INDEX = 1
TITLE_FONTSIZE = 16
AXES_FONTSIZE = 14

architectures = ["RNN-LN", "LSTM-LN", "RNN-LN-FW", "NTM2"]
"""
Get data.
"""
def get_saved_data_maxepochs(architecture, results_dir, template, trial_num):
    """Retrieve architecture results for the most train epochs.

    If no results file found for specified architecture and template, returns
    array of -1.
    Args:
        architecture: Name of the architecture.
        results_dir: Directory containing results saved by run_experiment.py.
        template: Format of saved results where template arguments are architecture
                  number of train epochs (ex. run_experiment.py result saving convention).

    Returns:
        max_epochs: Number of train epochs in results.
        results: De-serialized results.
    """
    max_epochs = 0
    for file in os.listdir(results_dir):
        if fnmatch.fnmatch(file, '%s_results_*trial%d*' % (architecture, trial_num)):
            file_epochs = int(file.split('epochs')[0].split('_')[-1])
            max_epochs = max(max_epochs, file_epochs)
    if max_epochs == 0:
        print("No results found for architecture %s" % architecture)
        return 0, [[-1],[-1]]
    with open(os.path.join(results_dir, template % (architecture, max_epochs, trial_num))) as f:
        return pickle.load(f)

def get_data(results_dir, template, trial_nums, epochs_dict, plot_index):
    """Retrieve results for all architectures.

    Args:
        results_dir: Directory containing results saved by run_experiment.py.
        template: Format of saved results where template arguments are architecture
                  number of train epochs (ex. run_experiment.py result saving convention).

    Returns:
        results: Dictionary of train epochs and results for each architecture.
    """
    results = {}
    print("architecture (num epochs): train_accuracy, test_accuracy")
    for architecture in architectures:
        results[architecture] = []
        for trial_num in trial_nums:
            results[architecture].append(get_saved_data_maxepochs(architecture, results_dir, template, trial_num)[plot_index][epochs_dict[architecture] - 1])
        print(architecture + ":" + str(results[architecture]))
    return results

def get_split_data(results_dir, template, trial_nums, epochs_dict, query):
    """Retrieve results for all architectures.

    Args:
        results_dir: Directory containing results saved by run_experiment.py.
        template: Format of saved results where template arguments are architecture
                  number of train epochs (ex. run_experiment.py result saving convention).

    Returns:
        results: Dictionary of train epochs and results for each architecture.
    """
    results = {}
    print("architecture (num epochs): train_accuracy, test_accuracy")
    for architecture in architectures:
        results[architecture] = {}
        for query in split_queries:
            results[architecture][query] = []
            for trial_num in trial_nums:
                results[architecture][query].append(get_saved_data_maxepochs(architecture, results_dir, template, trial_num)['accuracies'][query][epochs_dict[architecture] - 1])
    return results

"""
Create plots.
"""
def plot_data(data, epochs_dict, plot_index, experiment_title, experiment_title2, chance_rate, save=False, save_dir=None):
    """Generates a bar chart of overall accuracy.

    Args:
        data: {architecture:(train_epochs, results)} dictionary where results follows the conventions in run_experiment.py.
        epochs_dict: {architecture:epoch} dictionary of number of training epochs at which to get result for each architecture.
        plot_index: Index of results containing desired accuracies. (Based on run_experiment.py conventions, 0 for train and 1 for test accuracy.)
        experiment_title: Name of the experiment, which is used to title the plot.
        experiment_title2: Name of title, which is used in the second line of the plot.
        chance_rate: Chance accuracy rate.
        save, save_dir: If save=True, saves the plot to save_dir folder.
    """
    info_type_words = ["Train", "Test"]
    info_type = "%s Accuracy" % info_type_words[plot_index]
    width = 0.35
    bar_values = []
    bar_mins = []
    bar_maxs = []
    for architecture in architectures:
        bar_mean = np.mean(data[architecture])
        bar_values.append(bar_mean)
        bar_mins.append(bar_mean - np.min(data[architecture]))
        bar_maxs.append(np.max(data[architecture]) - bar_mean)
    xaxis_label = ["RNN\n%d epochs" % (epochs_dict["RNN-LN"]), "LSTM\n%d epochs" % (epochs_dict["LSTM-LN"]), "Fast Weights\n%d epochs" % (epochs_dict["RNN-LN-FW"]), "Reduced NTM\n%d epochs" % (epochs_dict["NTM2"])]
    ind = np.arange(len(xaxis_label))
    plt.bar(ind, bar_values, width, yerr=(bar_mins, bar_maxs), color='#0082c8')
    plt.axhline(y=chance_rate, label="Chance Rate", color='k', linestyle='--')
    plt.xticks(ind + width/2, xaxis_label)
    plt.ylabel(info_type, fontsize=AXES_FONTSIZE)
    plt.ylim([-0.01, 1.01])
    plt.title(" %s\n%s" % (experiment_title, experiment_title2), fontsize=TITLE_FONTSIZE)
    plt.legend(loc="upper left")
    if save:
        plt.savefig(os.path.join(save_dir, 'multitrial_barchart_%s%s_%s' % (experiment_title.replace("_","").replace(" ","").replace(",",""), experiment_title2.replace("_","").replace(" ","").replace(",",""), info_type.replace(" ",""))), bbox_inches="tight")
    # plt.show()
    plt.close()

def plot_data_double(data0, data1, label0, label1, epochs_dict, experiment_title, experiment_title2, chance_rate, save=False, save_dir=None, savename="both"):
    """Generates a bar chart of overall accuracy.

    Args:
        data: {architecture:(train_epochs, results)} dictionary where results follows the conventions in run_experiment.py.
        epochs_dict: {architecture:epoch} dictionary of number of training epochs at which to get result for each architecture.
        experiment_title: Name of the experiment, which is used to title the plot.
        experiment_title2: Name of title, which is used in the second line of the plot.
        chance_rate: Chance accuracy rate.
        save, save_dir: If save=True, saves the plot to save_dir folder.
    """
    width = 0.35
    xaxis_label = ["RNN\n%d epochs" % (epochs_dict["RNN-LN"]), "LSTM\n%d epochs" % (epochs_dict["LSTM-LN"]), "Fast Weights\n%d epochs" % (epochs_dict["RNN-LN-FW"]), "Reduced NTM\n%d epochs" % (epochs_dict["NTM2"])]
    ind = np.arange(len(xaxis_label))
    bar_values0 = []
    bar_mins0 = []
    bar_maxs0 = []
    for architecture in architectures:
        bar_mean0 = np.mean(data0[architecture])
        bar_values0.append(bar_mean0)
        bar_mins0.append(bar_mean0 - np.min(data0[architecture]))
        bar_maxs0.append(np.max(data0[architecture]) - bar_mean0)
    plt.bar(ind, bar_values0, width, color='#88b5ce', yerr=(bar_mins0, bar_maxs0), label=label0)
    bar_values1 = []
    bar_mins1 = []
    bar_maxs1 = []
    for architecture in architectures:
        bar_mean1 = np.mean(data1[architecture])
        bar_values1.append(bar_mean1)
        bar_mins1.append(bar_mean1 - np.min(data1[architecture]))
        bar_maxs1.append(np.max(data1[architecture]) - bar_mean1)
    plt.bar(ind+width, bar_values1, width, color='#0082c8', yerr=(bar_mins1, bar_maxs1), label=label1)
    plt.axhline(y=chance_rate, label="Chance Rate", color='k', linestyle='--')
    plt.xticks(ind+width/2, xaxis_label)
    plt.ylabel("Accuracy", fontsize=AXES_FONTSIZE)
    plt.ylim([-0.01, 1.01])
    plt.xlim([-0.25, 3.75])
    plt.title(" %s\n%s" % (experiment_title, experiment_title2), fontsize=TITLE_FONTSIZE)
    legend = plt.legend(ncol=3, bbox_to_anchor=(1.1, -0.1))
    if save:
        plt.savefig(os.path.join(save_dir, 'multitrial_barchart_%s%s_%s' % (experiment_title.replace("_","").replace(" ","").replace(",",""), experiment_title2.replace("_","").replace(" ","").replace(",",""), savename)), bbox_extra_artists=(legend,), bbox_inches="tight", dpi=600)
    # plt.show()
    plt.close()

def plot_split_data(data, epochs_dict, plot_index, experiment_title, experiment_title2, chance_rate, split_queries, save=False, save_dir=None):
    """Generates a bar chart of overall accuracy.

    Args:
        data: {architecture:(train_epochs, results)} dictionary where results follows the conventions in run_experiment.py.
        epochs_dict: {architecture:epoch} dictionary of number of training epochs at which to get result for each architecture.
        plot_index: Index of results containing desired accuracies. (Based on run_experiment.py conventions, 0 for train and 1 for test accuracy.)
        experiment_title: Name of the experiment, which is used to title the plot.
        experiment_title2: Name of title, which is used in the second line of the plot.
        chance_rate: Chance accuracy rate.
        split_queries: The order of queries saved in 'results', following the conventions in run_experiment.py.
        save, save_dir: If save=True, saves the plot to save_dir folder.
    """
    info_type_words = ["Train", "Test"]
    # query_colors = {"QDessert":"#72C1F3","QDrink":"#1498EB","QEmcee":"#0E67A0","QFriend":"#0A4C76","QPoet":"#062A42","QSubject":"#050a30"}
    #query_colors = {"QDessert":"#1f77b4", "QDrink":"#ff7f0e", "QEmcee":"#2ca02c", "QFriend":"#d62728", "QPoet":"#9467bd", "QSubject":"#8c564b"}
    query_colors = {"QDessert":"#006BA4", "QDrink":"#FF800E", "QEmcee":"#ABABAB", "QFriend":"#595959", "QPoet":"#5F9ED1", "QSubject":"#C85200"}
    info_type = "%s Accuracy" % info_type_words[plot_index]
    width = 0.3
    split_width = width/len(split_queries)
    xaxis_label = ["RNN\n%d epochs" % (epochs_dict["RNN-LN"]), "LSTM\n%d epochs" % (epochs_dict["LSTM-LN"]), "Fast Weights\n%d epochs" % (epochs_dict["RNN-LN-FW"]), "Reduced NTM\n%d epochs" % (epochs_dict["NTM2"])]
    xaxis_length = len(xaxis_label)
    for i in range(len(split_queries)):
        query = split_queries[i]
        bar_values = []
        bar_mins = []
        bar_maxs = []
        for architecture in architectures:
            bar_mean = np.mean(data[architecture][query])
            bar_values.append(bar_mean)
            bar_mins.append(bar_mean - np.min(data[architecture][query]))
            bar_maxs.append(np.max(data[architecture][query]) - bar_mean)
        xaxis_label = ["RNN\n%d epochs" % (epochs_dict["RNN-LN"]), "LSTM\n%d epochs" % (epochs_dict["LSTM-LN"]), "Fast Weights\n%d epochs" % (epochs_dict["RNN-LN-FW"]), "Reduced NTM\n%d epochs" % (epochs_dict["NTM2"])]
        ind = np.arange(len(xaxis_label))
        plt.bar(np.arange(xaxis_length) + split_width * i, bar_values, split_width, color=query_colors[query],  yerr=(bar_mins, bar_maxs), label=query, alpha=0.8)
    plt.axhline(y=chance_rate, label="Chance Rate", color='k', linestyle='--')
    ind = np.arange(xaxis_length)
    plt.xticks(ind + width/2, xaxis_label)
    plt.ylabel(info_type, fontsize=AXES_FONTSIZE)
    plt.ylim([-0.01, 1.01])
    plt.legend(loc="upper left")
    plt.title("%s\n%s" % (experiment_title, experiment_title2), fontsize=TITLE_FONTSIZE)
    # plt.show()
    if save:
        plt.savefig(os.path.join(save_dir, 'multitrial_barchart_%s%s_%s_split' % (experiment_title.replace("_","").replace(" ","").replace(",",""), experiment_title2.replace("_","").replace(" ","").replace(",",""), info_type.replace(" ",""))), bbox_inches="tight", dpi=600)
    plt.close()

if __name__ == '__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument('--exp_folder', help='Experiment folder containing results (after results/).', required=True)
    parser.add_argument('--exp_title', help='Experiment title (used in plots).', required=True)
    parser.add_argument('--exp_title2', help='Experiment title, second line (used in plots).', required=True)
    parser.add_argument('--trial_nums', help='Available trial nums.', type=str, required=True)
    parser.add_argument('--save', help='Whether to save plot.', choices=['True', 'False'], default='True')
    parser.add_argument('--split_queries', help='Split queries to plot.')
    parser.add_argument('--chance_rate', help='Chance accuracy rate.', type=float, required=True)
    parser.add_argument('--epochs_dict', help='Dictionary of epochs to plot.', required=True)
    parser.add_argument('--plot_style', help='Type of plot', choices=['traintest_separate', 'traintest_together'], default='traintest_separate')
    parser.add_argument('--plot_traintest_together', help='True if train and test results in same plot', choices=['True', 'False'], default='False')
    args=parser.parse_args()

    experiment_folder = args.exp_folder
    experiment_title = str(args.exp_title)
    experiment_title2 = str(args.exp_title2)
    trial_nums = ast.literal_eval(args.trial_nums)
    save = args.save == 'True'
    split_queries = args.split_queries
    chance_rate = args.chance_rate
    plot_style = args.plot_style == 'True'
    epochs_dict = ast.literal_eval(args.epochs_dict)
    results_dir = os.path.join(base_dir, "results", experiment_folder)
    save_dir = os.path.join(base_dir, "figures")
    plot_traintest_together = args.plot_traintest_together == 'True'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(trial_nums)
    SPLIT_TEMPLATE = "%s_results_%depochs_trial%d_split.p"
    COMBINED_TEMPLATE = "%s_results_%depochs_trial%d.p"

    if plot_traintest_together:
        # Plot train and test accuracy together.
        plot_index = 0
        data0 = get_data(results_dir, COMBINED_TEMPLATE, trial_nums, epochs_dict, plot_index)
        plot_index = 1
        data1 = get_data(results_dir, COMBINED_TEMPLATE, trial_nums, epochs_dict, plot_index)
        label0 = "Previously Seen Fillers"
        label1 = "Previously Unseen Fillers"
        plot_data_double(data0, data1, label0, label1, epochs_dict, experiment_title, experiment_title2, chance_rate, save, save_dir, savename="trainandtest")
    else:
        # Plot train and test accuracy separately.
        plot_index = 0
        data = get_data(results_dir, COMBINED_TEMPLATE, trial_nums, epochs_dict, plot_index)
        plot_data(data, epochs_dict, plot_index, experiment_title, experiment_title2, chance_rate, save, save_dir)
        plot_index = 1
        data = get_data(results_dir, COMBINED_TEMPLATE, trial_nums, epochs_dict, plot_index)
        plot_data(data, epochs_dict, plot_index, experiment_title, experiment_title2, chance_rate, save, save_dir)

    # Plot split test accuracy.
    if split_queries:
        split_queries = split_queries.split(",")
        split_data = get_split_data(results_dir, SPLIT_TEMPLATE, trial_nums, epochs_dict, split_queries)
        plot_split_data(split_data, epochs_dict, plot_index, experiment_title, experiment_title2, chance_rate, split_queries, save, save_dir)
