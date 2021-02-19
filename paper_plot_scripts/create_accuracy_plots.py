'''Generate accuracy plots and learning curves.'''
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

base_dir = '/home/cc27/Thesis/generalized_schema_learning'
plot_dir = os.path.join(base_dir, 'paper_plot_scripts', 'figures')
title_fontsize = 30
axis_fontsize = 16
yaxis_fontsize = 16
color_blue = '#1f77b4'
color_orange = '#FF800E'
query_colors = {"QDessert": "#006BA4", "QDrink": "#FF800E", "QEmcee": "#ABABAB", "QFriend": "#595959", "QPoet": "#5F9ED1", "QSubject": "#C85200"}
network_colors = {"RNN-LN":"#0082c8",
       "LSTM-LN":"#ffe119",
       "RNN-LN-FW":"#3cb44b",
       "DNC":"#000000"}
network_line_styles= {"RNN-LN":"solid",
       "LSTM-LN":"dotted",
       "RNN-LN-FW":"dashed",
       "DNC":"dashdot"}
queries = ['QDessert', 'QDrink', 'QEmcee', 'QFriend', 'QPoet', 'QSubject']
LINEWIDTH = 1
FILL_BETWEEN_ALPHA = 0.2


def accuracy_plots_fixed(trial_nums=[1, 2, 3]):
   # Figure 2.
   results_dir = os.path.join(base_dir, 'results/fixedfiller_AllQs/fixed_filler/fixed_filler')
   chance_rate = 0.02
   networks = ['RNN-LN', 'LSTM-LN', 'RNN-LN-FW', 'DNC']
   network_names = {
       'RNN-LN': 'RNN',
       'LSTM-LN': 'LSTM',
       'RNN-LN-FW': 'Fast Weights',
       'DNC': 'DNC'}
   epochs_dict = {
       'RNN-LN': 5000,
       'LSTM-LN': 1000,
       'RNN-LN-FW': 1000,
       'DNC': 50}
   titles = [network_names[network] for network in networks]
   template = '{network}_results_{num_epochs}epochs_trial{trial_num}.p'
   template_split = '{network}_results_{num_epochs}epochs_trial{trial_num}_split.p'
   for network_idx, network in enumerate(networks):
       network_vals = []
       for trial_num in trial_nums:
           with open(os.path.join(results_dir, template.format(network=network, num_epochs=epochs_dict[network], trial_num=trial_num)), 'rb') as f:
               results = pickle.load(f)
           with open(os.path.join(results_dir, template_split.format(network=network, num_epochs=epochs_dict[network], trial_num=trial_num)), 'rb') as f:
               split_results = pickle.load(f)
           print(split_results['accuracies']['unseen'][-1], trial_num, network)
           network_vals.append(results[1][-1])
       accuracies = np.array(network_vals)
       y_mean = np.mean(accuracies)
       y_min = np.min(accuracies)
       y_max = np.max(accuracies)
       y_err = [[y_mean - y_min], [y_max - y_mean]]
       plt.bar(network_idx + 1, y_mean, yerr=y_err, color=color_blue)
   plt.xticks(np.arange(1, len(networks) + 1), titles, fontsize=axis_fontsize)
   plt.ylabel('Test Accuracy', fontsize=axis_fontsize)
   plt.axhline(y=chance_rate, label="Chance Rate", color='k', linestyle='--')
   plt.legend(loc='lower right', bbox_to_anchor=(1, 1))

   plt.savefig(os.path.join(plot_dir, 'accuracies_fixed_filler'), dpi=600)
   plt.close()


def accuracy_plots_variable(trial_nums=[1, 2, 3]):
   # Figure 3.
   results_dir = os.path.join(base_dir, 'results/variablefiller_AllQs/variable_filler/variable_filler')
   chance_rate = 0.008
   networks = ['RNN-LN', 'LSTM-LN', 'RNN-LN-FW', 'DNC']
   network_names = {
       'RNN-LN': 'RNN',
       'LSTM-LN': 'LSTM',
       'RNN-LN-FW': 'Fast Weights',
       'DNC': 'DNC'}
   epochs_dict = {
       'RNN-LN': 30000,
       'LSTM-LN': 30000,
       'RNN-LN-FW': 30000,
       'DNC': 30000}
   titles = [network_names[network] for network in networks]
   template = '{network}_results_{num_epochs}epochs_trial{trial_num}.p'
   template_split = '{network}_results_{num_epochs}epochs_trial{trial_num}_split.p'
   template_one_query = 'predictions/test_analysis_results_{network}_{num_epochs}epochs_trial{trial_num}_test_{query}.p_noise0_zerovectornoiseFalse'
   for network_idx, network in enumerate(networks):
       network_vals = []
       for trial_num in trial_nums:
           with open(os.path.join(results_dir, template.format(network=network, num_epochs=epochs_dict[network], trial_num=trial_num)), 'rb') as f:
               results = pickle.load(f)
           with open(os.path.join(results_dir, template_split.format(network=network, num_epochs=epochs_dict[network], trial_num=trial_num)), 'rb') as f:
               split_results = pickle.load(f)
           print(split_results['accuracies']['unseen'][-1], trial_num, network)
           network_vals.append(results[1][-1])
       accuracies = np.array(network_vals)
       y_mean = np.mean(accuracies)
       y_min = np.min(accuracies)
       y_max = np.max(accuracies)
       y_err = [[y_mean - y_min], [y_max - y_mean]]
       plt.bar(network_idx + 1, y_mean, yerr=y_err, color=color_blue)
   plt.xticks(np.arange(1, len(networks) + 1), titles, fontsize=axis_fontsize)
   plt.ylabel('Test Accuracy', fontsize=axis_fontsize)
   plt.axhline(y=chance_rate, label="Chance Rate", color='k', linestyle='--')
   plt.legend(loc='lower right', bbox_to_anchor=(1, 1))

   plt.savefig(os.path.join(plot_dir, 'accuracies_variable_filler'), dpi=600)
   plt.close()

   for network_idx, network in enumerate(networks):
       network_vals = {query: [] for query in queries}
       for trial_num in trial_nums:
           for query in queries:
               with open(os.path.join(results_dir, template_split.format(network=network, num_epochs=epochs_dict[network], trial_num=trial_num)), 'rb') as f:
                   split_results = pickle.load(f)
               query_filename = query.upper()
               query_accuracy = split_results['accuracies'][query_filename][-1]
               network_vals[query].append(query_accuracy)
       means = []
       errs = []
       for query in queries:
           accuracies = np.array(network_vals[query])
           y_mean = np.mean(accuracies)
           y_min = np.min(accuracies)
           y_max = np.max(accuracies)
           y_err = [[y_mean - y_min], [y_max - y_mean]]
           means.append(y_mean)
           errs.append(y_err)
       x = network_idx + 1 + np.array([-0.25, -0.15, -0.05, 0.05, 0.15, 0.25])
       plt.bar(x, means, yerr=np.squeeze(np.array(errs)).T, color=[query_colors[query] for query in queries], width=0.1)
   plt.xticks(np.arange(1, len(networks) + 1), titles, fontsize=axis_fontsize)
   plt.ylabel('Test Accuracy', fontsize=axis_fontsize)
   plt.axhline(y=chance_rate, label="Chance Rate", color='k', linestyle='--')
   plt.savefig(os.path.join(plot_dir, 'accuracies_variable_filler_split'), dpi=600)
   plt.close()


def smooth(array, num_smooth=10):
   """Smooth the array over every num_smooth elements."""
   return np.mean(np.array(array[:len(array) - len(array) % num_smooth]).reshape(-1, num_smooth), axis=1)


def accuracy_learning_curves_fixed(trial_nums=[1, 2, 3]):
   # Supplementary.
   results_dir = os.path.join(base_dir, 'results/fixedfiller_AllQs/fixed_filler/fixed_filler')
   chance_rate = 0.02
   networks = ['RNN-LN', 'LSTM-LN', 'RNN-LN-FW', 'DNC']
   network_names = {
       'RNN-LN': 'RNN',
       'LSTM-LN': 'LSTM',
       'RNN-LN-FW': 'Fast Weights',
       'DNC': 'DNC'}
   epochs_dict = {
       'RNN-LN': 5000,
       'LSTM-LN': 1000,
       'RNN-LN-FW': 1000,
       'DNC': 50}
   colors_dict = {"RNN-LN":"#0082c8",
           "LSTM-LN":"#ffe119",
           "RNN-LN-FW":"#3cb44b",
           "DNC":"#000000"}
   template = '{network}_results_{num_epochs}epochs_trial{trial_num}.p'
   template_split = '{network}_results_{num_epochs}epochs_trial{trial_num}_split.p'
   for network_idx, network in enumerate(networks):
       for trial_num in trial_nums:
           with open(os.path.join(results_dir, template.format(network=network, num_epochs=epochs_dict[network], trial_num=trial_num)), 'rb') as f:
               results = pickle.load(f)
           with open(os.path.join(results_dir, template_split.format(network=network, num_epochs=epochs_dict[network], trial_num=trial_num)), 'rb') as f:
               split_results = pickle.load(f)
           if trial_num == trial_nums[0]:
               network_vals = np.expand_dims(np.array(results[1]), axis=0)
           else:
               network_vals = np.concatenate((network_vals, np.expand_dims(np.array(results[1]), axis=0)), axis=0)
       mean_smoothed = smooth(np.mean(network_vals, axis=0))
       max_smoothed = smooth(np.max(network_vals, axis=0))
       min_smoothed = smooth(np.min(network_vals, axis=0))
       color = network_colors[network]
       linestyle = network_line_styles[network]
       label = network_names[network]
       label = network_names[network]
       x = np.array(range(len(min_smoothed))) * 10
       plt.plot(x, mean_smoothed, label=label,  color=color, linewidth=LINEWIDTH)
       plt.plot(x, min_smoothed,  color=color, linewidth=LINEWIDTH, alpha=FILL_BETWEEN_ALPHA)
       plt.plot(x, max_smoothed, color=color, linewidth=LINEWIDTH, alpha=FILL_BETWEEN_ALPHA)
       plt.fill_between(x, min_smoothed, max_smoothed, color=color, alpha=FILL_BETWEEN_ALPHA)
   plt.axhline(y=chance_rate, label="Chance Rate", color='k', linestyle='--')
   plt.xlabel("Train epoch (smoothed over 10 epochs)", fontsize=axis_fontsize)
   plt.savefig(os.path.join(plot_dir, 'learning_curves_fixed_filler'), dpi=600)
   plt.legend(loc='upper left')
   plt.close()


def accuracy_learning_curves_variable(trial_nums=[1, 2, 3]):
   # Supplementary.
   results_dir = os.path.join(base_dir, 'results/variablefiller_AllQs/variable_filler/variable_filler')
   chance_rate = 0.008
   networks = ['RNN-LN', 'LSTM-LN', 'RNN-LN-FW', 'DNC']
   network_names = {
       'RNN-LN': 'RNN',
       'LSTM-LN': 'LSTM',
       'RNN-LN-FW': 'Fast Weights',
       'DNC': 'DNC'}
   epochs_dict = {
       'RNN-LN': 30000,
       'LSTM-LN': 30000,
       'RNN-LN-FW': 30000,
       'DNC': 30000}
   template = '{network}_results_{num_epochs}epochs_trial{trial_num}.p'
   template_split = '{network}_results_{num_epochs}epochs_trial{trial_num}_split.p'
   for network_idx, network in enumerate(networks):
       for trial_num in trial_nums:
           with open(os.path.join(results_dir, template.format(network=network, num_epochs=epochs_dict[network], trial_num=trial_num)), 'rb') as f:
               results = pickle.load(f)
           with open(os.path.join(results_dir, template_split.format(network=network, num_epochs=epochs_dict[network], trial_num=trial_num)), 'rb') as f:
               split_results = pickle.load(f)
           if trial_num == trial_nums[0]:
               network_vals = np.expand_dims(np.array(results[1]), axis=0)
           else:
               network_vals = np.concatenate((network_vals, np.expand_dims(np.array(results[1]), axis=0)), axis=0)
       mean_smoothed = smooth(np.mean(network_vals, axis=0))
       max_smoothed = smooth(np.max(network_vals, axis=0))
       min_smoothed = smooth(np.min(network_vals, axis=0))
       color = network_colors[network]
       linestyle = network_line_styles[network]
       label = network_names[network]
       x = np.array(range(len(min_smoothed))) * 10
       plt.plot(x, mean_smoothed,  label=label, color=color, linewidth=LINEWIDTH)
       plt.plot(x, min_smoothed, color=color, linewidth=LINEWIDTH, alpha=FILL_BETWEEN_ALPHA)
       plt.plot(x, max_smoothed, color=color, linewidth=LINEWIDTH, alpha=FILL_BETWEEN_ALPHA)
       plt.fill_between(x, min_smoothed, max_smoothed, color=color, alpha=FILL_BETWEEN_ALPHA)
   plt.axhline(y=chance_rate, label="Chance Rate", color='k', linestyle='--')
   plt.xlabel("Train epoch (smoothed over 10 epochs)", fontsize=axis_fontsize)
   legend = plt.legend(loc='lower left', bbox_to_anchor=(0, 1.05), ncol=5)
   plt.savefig(os.path.join(plot_dir, 'learning_curves_variable_filler'), dpi=600)
   plt.close()

if __name__ == '__main__':
   accuracy_plots_fixed()
   accuracy_plots_variable()
   accuracy_learning_curves_fixed()
   accuracy_learning_curves_variable()
