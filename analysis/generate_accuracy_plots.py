"""Modules to generate custom plots.

Assumes results stored using train module in run_experiment.py.
"""
import argparse
import fnmatch
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

"""
Get data.
"""
def get_saved_data_maxepochs(architecture, results_dir, template):
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
        if fnmatch.fnmatch(file, '%s_results_*' % architecture):
            file_epochs = int(file.split('epochs')[0].split('_')[-1])
            max_epochs = max(max_epochs, file_epochs)
    if max_epochs == 0:
        print("No results found for architecture %s" % architecture)
        return 0, [[-1],[-1]]
    with open(os.path.join(results_dir, template % (architecture, max_epochs))) as f:
        return max_epochs, pickle.load(f)

def get_data(results_dir, template):
    """Retrieve results for all architectures.

    Args:
        results_dir: Directory containing results saved by run_experiment.py.
        template: Format of saved results where template arguments are architecture
                  number of train epochs (ex. run_experiment.py result saving convention).

    Returns:
        results: Dictionary of train epochs and results for each architecture.
    """
    architectures = ["RNN-LN-FW", "LSTM-LN", "NTM2", "RNN-LN"]
    results = {}
    print("architecture (num epochs): train_accuracy, test_accuracy")
    for architecture in architectures:
        results[architecture] = get_saved_data_maxepochs(architecture, results_dir, template)
        if "split" not in template:
            print("%s (%d): %f, %f" % (architecture, results[architecture][0], results[architecture][1][TRAIN_INDEX][-1], results[architecture][1][TEST_INDEX][-1]))
    return results

def smooth(array, num_smooth):
    """Smooth the array over every num_smooth elements."""
    return np.mean(np.array(array).reshape(-1,num_smooth), axis=1)

def get_smoothed_accuracies(data, plot_index, num_smooth):
    """Retrieve accuracies for each architecture."""
    rnnlnfw_smoothed = smooth(data['RNN-LN-FW'][1][plot_index], num_smooth)
    lstmln_smoothed = smooth(data['LSTM-LN'][1][plot_index], num_smooth)
    ntm2_smoothed = smooth(data['NTM2'][1][plot_index], num_smooth)
    rnnln_smoothed = smooth(data['RNN-LN'][1][plot_index], num_smooth)
    return rnnlnfw_smoothed, lstmln_smoothed, ntm2_smoothed, rnnln_smoothed

def get_smoothed_accuracies_split(data, query, num_smooth):
    """Retrieve accuracies for each architecture."""
    rnnlnfw_smoothed = smooth(data['RNN-LN-FW'][1]['accuracies'][query], num_smooth)
    lstmln_smoothed = smooth(data['LSTM-LN'][1]['accuracies'][query], num_smooth)
    ntm2_smoothed = smooth(data['NTM2'][1]['accuracies'][query], num_smooth)
    rnnln_smoothed = smooth(data['RNN-LN'][1]['accuracies'][query], num_smooth)
    return rnnlnfw_smoothed, lstmln_smoothed, ntm2_smoothed, rnnln_smoothed

"""
Create plots.
"""
def plot_data(data, plot_index, experiment_title, num_smooth=1, legend=False, xlim=None, ylim=None, save=False, save_dir=None):
    """Generates a plot of overall accuracy.

    Args:
        data: {architecture:(train_epochs, results)} dictionary where results follows the conventions in run_experiment.py.
        plot_index: Index of results containing desired accuracies. (Based on run_experiment.py conventions, 0 for train and 1 for test accuracy.)
        experiment_title: Name of the experiment, which is used to title the plot.
        num_smooth: Number of accuracies over which to smooth.
        xlim: Upper limit on the x-axis.
        save, save_dir: If save=True, saves the plot to save_dir folder.
    """
    rnnlnfw_smoothed, lstmln_smoothed, ntm2_smoothed, rnnln_smoothed = get_smoothed_accuracies(data, plot_index, num_smooth)
    info_type_words = ["Train", "Test"]
    info_type = "%s Accuracy" % info_type_words[plot_index]
    plt.plot(rnnlnfw_smoothed, label="Fast Weights", color='#3cb44b')
    plt.plot(lstmln_smoothed, label="LSTM", color='#ffe119')
    plt.plot(ntm2_smoothed, label="Reduced NTM", color='#000000')
    plt.plot(rnnln_smoothed, label="RNN", color='#0082c8')
    if legend:
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    if num_smooth > 1:
        plt.xlabel("Train epoch (smoothed over %d epochs)" % (num_smooth), fontsize=AXES_FONTSIZE)
    else:
        plt.xlabel("Train epoch", fontsize=AXES_FONTSIZE)
    plt.ylabel(info_type.title(), fontsize=AXES_FONTSIZE)
    plt.gca().set_xlim(left=0)
    if ylim:
        plt.ylim(ylim)
    else:
        plt.gca().set_ylim(bottom=0)
    if xlim:
        plt.xlim([0, xlim / num_smooth])
    xtick_labels = [str(int(i * num_smooth)) for i in plt.xticks()[0]]
    plt.xticks(plt.xticks()[0], xtick_labels)
    plt.title("%s for %s" % (info_type, experiment_title), fontsize=TITLE_FONTSIZE)
    if save:
        plt.savefig(os.path.join(save_dir, '%s_%s_smoothed%d' % (experiment_title.replace("_","").replace("\n","").replace(" ","").replace(",",""), info_type.replace(" ",""), num_smooth)), bbox_inches="tight")
    plt.show()

def plot_split_data(data, experiment_title, num_smooth, split_queries, xlim=None, save=False, save_dir=None):
    """Generates a plot of accuracies, split by query.

    Args:
        data: A {architecture:(train_epochs, results)} dictionary where 'results' contains results split by query, following the conventions in run_experiment.py.
        experiment_title: The name of the experiment, which is used to title the plot.
        num_smooth: The number of accuracies over which to smooth.
        split_queries: The order of queries saved in 'results', following the conventions in run_experiment.py.
        xlim: The upper limit on the x-axis.
        save, save_dir: If save=True, saves the plot to save_dir folder.
    """
    fig, axes = plt.subplots(len(split_queries), 1, sharex='col')
    fig.suptitle("Test Accuracies for %s" % (experiment_title), fontsize=TITLE_FONTSIZE)
    for i in range(len(split_queries)):
        axes[i].tick_params(axis='both')
        axes[i].set_ylabel(split_queries[i], fontsize=AXES_FONTSIZE-5)
        query = split_queries[i]
        rnnlnfw_smoothed, lstmln_smoothed, ntm2_smoothed, rnnln_smoothed = get_smoothed_accuracies_split(data, query, num_smooth)
        axes[i].plot(rnnlnfw_smoothed, label="Fast Weights", color='#3cb44b')
        axes[i].plot(lstmln_smoothed, label="LSTM", color='#ffe119')
        axes[i].plot(ntm2_smoothed, label="NTM2", color='#000000')
        axes[i].plot(rnnln_smoothed, label="RNN", color='#0082c8')
    axes[0].legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    if num_smooth > 1:
        plt.xlabel("Train epoch (smoothed over %d epochs)" % (num_smooth), fontsize=AXES_FONTSIZE)
    else:
        plt.xlabel("Train epoch.", fontsize=AXES_FONTSIZE)
    if xlim:
        plt.xlim([0, xlim / num_smooth])
    plt.gca().set_xlim(left=0)
    xtick_labels = [str(int(j * num_smooth)) for j in axes[0].get_xticks()]
    axes[0].set_xticklabels(xtick_labels)
    if save:
        plt.savefig(os.path.join(save_dir, '%s_test_smoothed%d_split' % (experiment_title.replace("_","").replace("\n","").replace(" ","").replace(",",""), num_smooth)), bbox_inches="tight")
    plt.show()

def plot_curriculum_learning(experiment_title, combined_file, split_file, first_regime_num, num_smooth, legend=True, xlim=None, save=False, save_dir=None):
    """Generates a plot of subject, poet, and overall accuracies.

    Hard-coded indices for subject-poet curriculum learning.

    Args:
        experiment_title: The name of the experiment, which is used to title the plot.
        combined_file: File containing overall results, following conventions in run_experiment.py.
        split_file: File containing results split by query, following conventions in run_experiment.py.
        first_regime_num: String describing number of epochs in first regime.
        num_smooth: The number of accuracies over which to smooth.
        save, save_dir: If save=True, saves the plot to save_dir folder.
    """
    TEST_INDEX = 1
    SUBJECT_INDEX = 0
    POET_INDEX = 2
    with open(combined_file) as f:
        combined_results = pickle.load(f)
    with open(split_file) as f:
        split_results = pickle.load(f)

    plt.plot(smooth(combined_results[TEST_INDEX], num_smooth),label="Ovarall Accuracy")
    plt.plot(smooth(split_results[SUBJECT_INDEX], num_smooth),label="Subject Accuracy")
    plt.plot(smooth(split_results[POET_INDEX], num_smooth),label="Poet Accuracy")
    plt.title(experiment_title, fontsize=TITLE_FONTSIZE)
    plt.xlabel("Train Epochs (%s in first regime, smoothed over %d epochs)" % (first_regime_num, num_smooth), fontsize=AXES_FONTSIZE)
    plt.ylabel("Test Accuracy", fontsize=AXES_FONTSIZE)
    if xlim:
        plt.xlim([0, xlim / num_smooth])
    xtick_labels = [str(int(i * num_smooth)) for i in plt.xticks()[0]]
    plt.xticks(plt.xticks()[0], xtick_labels)
    if legend:
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    plt.gca().set_xlim(left=0)
    if save:
        savename = experiment_title.replace(" ","").replace(",","")
        plt.savefig(save_dir + "%s_smoothed%d" % (savename, num_smooth), bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument('--exp_folder', help='Experiment folder containing results (after results/).', required=True)
    parser.add_argument('--exp_title', help='Experiment title (used in plots).', required=True)
    parser.add_argument('--trial_num', help='Number of trial.', type=int, required=True)
    parser.add_argument('--num_smooth', help='Window size for smoothing.', type=int, default=10)
    parse.add_argument('--save', help='Whether to save plot.', choices=['True', 'False'], default='True')
    parse.add_argument('--legend', help='Whether include legend in plot.', choices=['True', 'False'], default='True')
    args=parser.parse_args()

    experiment_folder = args.exp_folder
    experiment_title = args.exp_title
    trial_num = args.trial_num
    num_smooth = args.num_smooth
    save = args.save == 'True'
    legend = args.legend == 'True'

    xlim = None
    ylim = [0, 1.03]
    results_dir = os.path.join(base_dir, "results", experiment_folder)
    save_dir = os.path.join(base_dir, "figures")
    SPLIT_TEMPLATE = "%s_results_%depochs_" + "trial%d_split.p" % trial_num
    COMBINED_TEMPLATE = "%s_results_%depochs_" + "trial%d.p" % trial_num

    # Plot train accuracy.
    ylim = [-0.05, 1.03]
    plot_index = 0
    results = get_data(results_dir, COMBINED_TEMPLATE)
    plot_data(results, plot_index, experiment_title, num_smooth, legend=legend, xlim=xlim, ylim=ylim, save=save, save_dir=save_dir)

    # Plot test accuracy.
    ylim = [-0.05, 1.03]
    plot_index = 1
    results = get_data(results_dir, COMBINED_TEMPLATE)
    plot_data(results, plot_index, experiment_title, num_smooth, legend=legend, xlim=xlim, ylim=ylim, save=save, save_dir=save_dir)

    # Plot split test accuracy.
    results = get_data(results_dir, SPLIT_TEMPLATE)
    plot_split_data(results, experiment_title, num_smooth, split_queries, xlim=xlim, save=save, save_dir=save_dir)
