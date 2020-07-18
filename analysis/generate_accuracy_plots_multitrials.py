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
LINEWIDTH = 1
FILL_BETWEEN_ALPHA = 0.2

architectures = ["RNN-LN", "LSTM-LN", "RNN-LN-FW", "NTM2"]
COLORS = {"RNN-LN":"#0082c8", "LSTM-LN":"#ffe119", "RNN-LN-FW":"#3cb44b", "NTM2":"#000000"}
LABELS = {"RNN-LN":"RNN", "LSTM-LN":"LSTM", "RNN-LN-FW":"Fastweights", "NTM2":"Reduced NTM"}
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
        return max_epochs, pickle.load(f)

def get_data(results_dir, template, trial_nums, plot_index):
    """Retrieve results for all architectures.

    Args:
        results_dir: Directory containing results saved by run_experiment.py.
        template: Format of saved results where template arguments are architecture
                  number of train epochs (ex. run_experiment.py result saving convention).

    Returns:
        results: Dictionary with numpy array for each architecture.
                 Array rows are max, min, mean error for each epoch.
    """
    num_trial_nums = len(trial_nums)
    results = {}
    for architecture in architectures:
        max_epochs = []
        all_trial_accuracies = []
        for trial_num in trial_nums:
            trial_max_epochs, trial_accuracies = get_saved_data_maxepochs(architecture, results_dir, template, trial_num)
            trial_accuracies = trial_accuracies[plot_index]
            max_epochs.append(trial_max_epochs)
            all_trial_accuracies.append(trial_accuracies)
        min_max_epochs = min(max_epochs)
        all_trial_accuracies_array = np.empty((num_trial_nums, min_max_epochs))
        for i in range(num_trial_nums):
            all_trial_accuracies_array[i] = all_trial_accuracies[i][:min_max_epochs]
        architecture_results = {}
        architecture_results['min'] = np.min(all_trial_accuracies_array, axis=0)
        architecture_results['max'] = np.max(all_trial_accuracies_array, axis=0)
        architecture_results['mean'] = np.mean(all_trial_accuracies_array, axis=0)
        results[architecture] = architecture_results
    return results

def get_data_query(results_dir, template, trial_nums, query):
    """Retrieve results for all architectures.

    Args:
        results_dir: Directory containing results saved by run_experiment.py.
        template: Format of saved results where template arguments are architecture
                  number of train epochs (ex. run_experiment.py result saving convention).

    Returns:
        results: Dictionary with numpy array for each architecture.
                 Array rows are max, min, mean error for each epoch.
    """
    num_trial_nums = len(trial_nums)
    results = {}
    for architecture in architectures:
        max_epochs = []
        all_trial_accuracies = []
        for trial_num in trial_nums:
            trial_max_epochs, trial_accuracies = get_saved_data_maxepochs(architecture, results_dir, template, trial_num)
            max_epochs.append(trial_max_epochs)
            all_trial_accuracies.append(trial_accuracies["accuracies"][query])
        min_max_epochs = min(max_epochs)
        all_trial_accuracies_array = np.empty((num_trial_nums, min_max_epochs))
        for i in range(num_trial_nums):
            all_trial_accuracies_array[i] = all_trial_accuracies[i][:min_max_epochs]
        architecture_results = {}
        architecture_results['min'] = np.min(all_trial_accuracies_array, axis=0)
        architecture_results['max'] = np.max(all_trial_accuracies_array, axis=0)
        architecture_results['mean'] = np.mean(all_trial_accuracies_array, axis=0)
        results[architecture] = architecture_results
    return results

def smooth(array, num_smooth):
    """Smooth the array over every num_smooth elements."""
    return np.mean(np.array(array[:len(array)-len(array)%num_smooth]).reshape(-1,num_smooth), axis=1)

"""
Create plots.
"""
def plot_data(data, plot_index, experiment_title, experiment_title2, chance_rate, num_smooth=1, legend=False, xlim=None, ylim=None, save=False, save_dir=None):
    """Generates a plot of overall accuracy.

    Args:
        data: {architecture:(train_epochs, results)} dictionary where results follows the conventions in run_experiment.py.
        plot_index: Index of results containing desired accuracies. (Based on run_experiment.py conventions, 0 for train and 1 for test accuracy.)
        experiment_title: Name of the experiment, which is used to title the plot.
        experiment_title2: Name of title, which is used in the second line of the plot.
        num_smooth: Number of accuracies over which to smooth.
        xlim: Upper limit on the x-axis.
        save, save_dir: If save=True, saves the plot to save_dir folder.
    """
    info_type_words = ["Train", "Test"]
    info_type = "%s Accuracy" % info_type_words[plot_index]
    for architecture in architectures:
        color = COLORS[architecture]
        label = LABELS[architecture]
        mean_smoothed = smooth(data[architecture]["mean"], num_smooth)
        min_smoothed = smooth(data[architecture]["min"], num_smooth)
        max_smoothed = smooth(data[architecture]["max"], num_smooth)
        x = range(len(min_smoothed))
        plt.plot(x, mean_smoothed, label=label, color=color, linewidth=LINEWIDTH)
        plt.plot(x, min_smoothed, color=color, linewidth=LINEWIDTH, alpha=FILL_BETWEEN_ALPHA)
        plt.plot(x, max_smoothed, color=color, linewidth=LINEWIDTH, alpha=FILL_BETWEEN_ALPHA)
        plt.fill_between(x, min_smoothed, max_smoothed, color=color, alpha=FILL_BETWEEN_ALPHA)
    plt.axhline(y=chance_rate, label="Chance Rate", color='k', linestyle='--')
    if legend:
        legend = plt.legend(loc="upper left", bbox_to_anchor=(1.03, 1.01))
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
    plt.title("%s for %s\n%s" % (info_type, experiment_title, experiment_title2), fontsize=TITLE_FONTSIZE)
    # plt.show()
    if save:
        plt.savefig(os.path.join(save_dir, '%s%s_%s_smoothed%d_withribbons' % (experiment_title.replace("_","").replace(" ","").replace(",",""), experiment_title2.replace("_","").replace(" ","").replace(",",""),
         info_type.replace(" ",""), num_smooth)), bbox_inches="tight")
    plt.close()

def plot_split_data(results_dir, template, trial_nums, experiment_title, experiment_title2, num_smooth, split_queries, chance_rate, xlim=None, ylim=None, save=False, save_dir=None):
    """Generates a plot of accuracies, split by query.

    Args:
        data: A {architecture:(train_epochs, results)} dictionary where 'results' contains results split by query, following the conventions in run_experiment.py.
        experiment_title: The name of the experiment, which is used to title the plot.
        experiment_title2: Name of title, which is used in the second line of the plot.
        num_smooth: The number of accuracies over which to smooth.
        split_queries: The order of queries saved in 'results', following the conventions in run_experiment.py.
        xlim: The upper limit on the x-axis.
        save, save_dir: If save=True, saves the plot to save_dir folder.
    """
    fig, axes = plt.subplots(len(split_queries), 1, sharex='col')
    fig.suptitle("Test Accuracies for %s\n%s" % (experiment_title, experiment_title2), fontsize=TITLE_FONTSIZE)
    for i in range(len(split_queries)):
        query = split_queries[i]
        axes[i].tick_params(axis='both')
        axes[i].set_ylabel(split_queries[i], fontsize=AXES_FONTSIZE-8)
        data = get_data_query(results_dir, template, trial_nums, query)
        for architecture in architectures:
            color = COLORS[architecture]
            label = LABELS[architecture]
            mean_smoothed = smooth(data[architecture]["mean"], num_smooth)
            min_smoothed = smooth(data[architecture]["min"], num_smooth)
            max_smoothed = smooth(data[architecture]["max"], num_smooth)
            x = range(len(min_smoothed))
            axes[i].plot(x, mean_smoothed, label=label, color=color, linewidth=LINEWIDTH)
            axes[i].plot(x, min_smoothed, color=color, linewidth=LINEWIDTH, alpha=FILL_BETWEEN_ALPHA)
            axes[i].plot(x, max_smoothed, color=color, linewidth=LINEWIDTH, alpha=FILL_BETWEEN_ALPHA)
            axes[i].fill_between(x, min_smoothed, max_smoothed, color=color, alpha=FILL_BETWEEN_ALPHA)
        axes[i].axhline(y=chance_rate, label="Chance Rate", color='k', linestyle='--')
    # axes[0].legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    if num_smooth > 1:
        plt.xlabel("Train epoch (smoothed over %d epochs)" % (num_smooth), fontsize=AXES_FONTSIZE)
    else:
        plt.xlabel("Train epoch.", fontsize=AXES_FONTSIZE)
    if xlim:
        plt.xlim([0, xlim / num_smooth])
    if ylim:
        plt.ylim(ylim)
    plt.gca().set_xlim(left=0)
    xtick_labels = [str(int(j * num_smooth)) for j in axes[0].get_xticks()]
    axes[0].set_xticklabels(xtick_labels)
    # plt.show()
    if save:
        plt.savefig(os.path.join(save_dir, '%s%s_TestAccuracy_smoothed%d_split_withribbons' % (experiment_title.replace("_","").replace(" ","").replace(",",""), experiment_title2.replace("_","").replace(" ","").replace(",",""), num_smooth)), bbox_inches="tight")
    plt.close()

if __name__ == '__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument('--exp_folder', help='Experiment folder containing results (after results/).', required=True)
    parser.add_argument('--exp_title', help='Experiment title (used in plots).', required=True)
    parser.add_argument('--exp_title2', help='Experiment title, second line (used in plots).', required=True)
    parser.add_argument('--trial_nums', help='Available trial nums.', type=str, required=True)
    parser.add_argument('--num_smooth', help='Window size for smoothing.', type=int, default=10)
    parser.add_argument('--save', help='Whether to save plot.', choices=['True', 'False'], default='True')
    parser.add_argument('--legend', help='Whether include legend in plot.', choices=['True', 'False'], default='True')
    parser.add_argument('--split_queries', help='Split queries to plot.')
    parser.add_argument('--chance_rate', help='Chance accuracy rate.', type=float, required=True)
    parser.add_argument('--xlim', help='x-axis maximum limit.')
    args=parser.parse_args()

    experiment_folder = args.exp_folder
    experiment_title = str(args.exp_title)
    experiment_title2 = str(args.exp_title2)
    trial_nums = ast.literal_eval(args.trial_nums)
    num_smooth = int(args.num_smooth)
    save = args.save == 'True'
    legend = args.legend == 'True'
    split_queries = args.split_queries
    chance_rate = args.chance_rate
    xlim = int(args.xlim)
    ylim = [0, 1.03]
    results_dir = os.path.join(base_dir, "results", experiment_folder)
    save_dir = os.path.join(base_dir, "figures")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    SPLIT_TEMPLATE = "%s_results_%depochs_trial%d_split.p"
    COMBINED_TEMPLATE = "%s_results_%depochs_trial%d.p"

    # Plot train accuracy.
    plot_index = 0
    results = get_data(results_dir, COMBINED_TEMPLATE, trial_nums, plot_index)
    plot_data(results, plot_index, experiment_title, experiment_title2, chance_rate, num_smooth, legend=legend, xlim=xlim, ylim=ylim, save=save, save_dir=save_dir)

    # Plot test accuracy.
    plot_index = 1
    results = get_data(results_dir, COMBINED_TEMPLATE, trial_nums, plot_index)
    plot_data(results, plot_index, experiment_title, experiment_title2, chance_rate, num_smooth, legend=legend, xlim=xlim, ylim=ylim, save=save, save_dir=save_dir)

    # Plot split test accuracy.
    if split_queries:
        split_queries = split_queries.split(",")
        plot_split_data(results_dir, SPLIT_TEMPLATE, trial_nums, experiment_title, experiment_title2, num_smooth, split_queries, chance_rate, xlim=xlim, ylim=ylim, save=save, save_dir=save_dir)
