import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

from scipy import stats

base_dir = '/home/cc27/Thesis/generalized_schema_learning'
plot_dir = os.path.join(base_dir, 'paper_plot_scripts', 'figures')
distributions_results_dir = os.path.join(base_dir, 'results/variablefiller_AllQs/variable_filler_distributions')

color_blue = '#1f77b4'
color_orange = '#FF800E'
axis_fontsize = 16
distributions_dict = {
    'qsubject': 'B',
    'qfriend': 'B',
    'qdrink': 'B',
    'qdessert': 'A',
    'qemcee': 'A',
    'qpoet': 'A'}

query_to_test_name = {'qdessert': 'QDESSERT',
        'qemcee': 'QEMCEE',
        'qpoet': 'QPOET',
        'qsubject': 'QSUBJECT',
        'qfriend': 'QFRIEND',
        'qdrink': 'QDRINK'}

colors_dict = {query: color_blue if distributions_dict[query] == 'B' else color_orange for query in distributions_dict}

queries = ['qdessert', 'qemcee', 'qpoet', 'qsubject', 'qfriend', 'qdrink']

network_names_dict = {
    'RNN-LN-FW': 'Fast Weights',
    'DNC': 'DNC'}

title_fontsize = 20
axis_fontsize = 12
yaxis_fontsize = 10
alpha = 0.4


def plot_hists(network_names=['DNC', 'RNN-LN-FW'],
        hist_type='values'):
    with open('logit_pvalues.p', 'rb') as f:
        results_df = pickle.load(f)

    bins = np.arange(-0.2, 0.2, 0.01)
    for network_name in network_names:
        print(network_name)
        results_df_network = results_df[results_df['network_name'] == network_name]
        fig, axes = plt.subplots(len(queries), 1)
        for idx, query in enumerate(queries):
            query_even_logits = []
            query_odd_logits = []
            for trial_idx, trial_results in results_df_network[results_df_network['query'] == query].iterrows():
                query_even_logits += list(trial_results['even_logits'])
                query_odd_logits += list(trial_results['odd_logits'])
            axes[idx].hist(query_odd_logits, label='odd', bins=bins, color=color_blue, alpha=alpha)
            axes[idx].hist(query_even_logits, label='even', bins=bins, color=color_orange, alpha=alpha)
            axes[idx].set_ylabel(query[:2].upper() + query[2:].lower(), color=colors_dict[query], fontsize=yaxis_fontsize)
            if idx < len(queries) - 1:
                axes[idx].set_xticks([])
        axes[0].legend()
        axes[0].set_title(network_names_dict[network_name], fontsize=title_fontsize)
        plt.savefig('figures/hist_{hist_type}_{network_name}'.format(hist_type=hist_type, network_name=network_name))
        plt.close()


def plot_bias_ttest_diffs(results_dir='variable_filler_distributions_all_randn_distribution',
        network_names=['DNC', 'RNN-LN-FW'],
        trial_nums=[10, 11, 12, 13, 14]):
    results_df = pd.DataFrame(columns=['network_name', 'trial_num', 'ttest_stat', 'pvalue', 'query'])
    for trial_num in trial_nums:
        for network_name in network_names:
            for query in query_to_test_name:
                logits = np.load(os.path.join(distributions_results_dir, '{results_dir}/predictions/logits_{network_name}_30000epochs_trial{trial_num}_test_{query_test_name}.p_noise0_zerovectornoiseFalse.npz'.format(network_name=network_name, trial_num=trial_num, query_test_name=query_to_test_name[query], results_dir=results_dir)))
                logits_pred = logits['predicted_logits']
                logits_true = logits['true_logits']
                logits_diff = logits_pred - logits_true

                s, pvalue = stats.ttest_ind(logits_diff[:, ::2].ravel(), logits_diff[:, 1::2].ravel())
                results_df = results_df.append({'network_name': network_name, 'trial_num': trial_num, 'ttest_stat': s, 'pvalue': pvalue, 'query': query, 'even_logits': logits_diff[:, ::2].ravel(), 'odd_logits': logits_diff[:, 1::2].ravel()}, ignore_index=True)

    query_cat = pd.Categorical(results_df['query'], categories=query_to_test_name.keys())
    results_df = results_df.assign(query_cat=query_cat)
    results_df.to_pickle('logit_pvalues_diffs.p')
    for network_name in network_names:
        results_df_network = results_df[results_df['network_name'] == network_name]
        for query_index, query in enumerate(queries):
            x = query_index + 1 + np.array([-0.2, -0.1, 0, 0.1, 0.2])
            y = results_df_network[results_df_network['query'] == query]['ttest_stat']
            plt.bar(x, y, width=0.09, color=colors_dict[query])
        plt.title(network_names_dict[network_name], fontsize=title_fontsize)
        plt.ylabel('t-Statistic', fontsize=axis_fontsize)
        query_names = [query[:2].upper() + query[2:].lower() for query in queries]
        plt.xticks(np.arange(1, len(queries) + 1), query_names, fontsize=axis_fontsize)
        for ticklabel in plt.gca().get_xticklabels():
            ticklabel.set_color(colors_dict[ticklabel.get_text().lower()])
        plt.savefig('figures/ttest_diffs_{network_name}_{results_dir}'.format(network_name=network_name, results_dir=results_dir))
        plt.close()


def plot_bias_ttest(results_dir='variable_filler_distributions_5050_AB',
        network_names=['DNC', 'RNN-LN-FW'],
        trial_nums=[1, 2, 3, 4, 5]):
    results_df = pd.DataFrame(columns=['network_name', 'trial_num', 'ttest_stat', 'pvalue', 'query'])
    queries = ['qdessert', 'qemcee', 'qpoet', 'qsubject', 'qfriend', 'qdrink']
    query_to_test_name = {'qdessert': 'QDESSERT',
            'qemcee': 'QEMCEE',
            'qpoet': 'QPOET',
            'qsubject': 'QSUBJECT',
            'qfriend': 'QFRIEND',
            'qdrink': 'QDRINK'}
    for trial_num in trial_nums:
        for network_name in network_names:
            for query in query_to_test_name:
                query_logits = np.load(os.path.join(distributions_results_dir, '{results_dir}/predictions/logits_{network_name}_30000epochs_trial{trial_num}_test_{query_test_name}.p_noise1_zerovectornoiseTrue.npz'.format(results_dir=results_dir, network_name=network_name, trial_num=trial_num, query_test_name=query_to_test_name[query])))['predicted_logits']
                s, pvalue = stats.ttest_ind(query_logits[:, ::2].ravel(), query_logits[:, 1::2].ravel())
                results_df = results_df.append({'network_name': network_name, 'trial_num': trial_num, 'ttest_stat': s, 'pvalue': pvalue, 'query': query.lower(), 'even_logits': query_logits[:, ::2].ravel(), 'odd_logits': query_logits[:, 1::2].ravel()}, ignore_index=True)

    query_cat = pd.Categorical(results_df['query'], categories=queries)
    results_df = results_df.assign(query_cat=query_cat)
    results_df.to_pickle('logit_pvalues.p')
    for network_name in network_names:
        results_df_network = results_df[results_df['network_name'] == network_name]
        for query_index, query in enumerate(queries):
            x = query_index + 1 + np.array([-0.2, -0.1, 0, 0.1, 0.2])
            y = results_df_network[results_df_network['query'] == query]['ttest_stat']
            plt.bar(x, y, width=0.09, color=colors_dict[query])
        plt.title(network_names_dict[network_name], fontsize=title_fontsize)
        plt.ylabel('t-Statistic', fontsize=axis_fontsize)
        query_names = [query[:2].upper() + query[2:].lower() for query in queries]
        plt.xticks(np.arange(1, len(queries) + 1), query_names, fontsize=axis_fontsize)
        for ticklabel in plt.gca().get_xticklabels():
            ticklabel.set_color(colors_dict[ticklabel.get_text().lower()])
        plt.savefig('figures/ttest_{network_name}_{results_dir}'.format(network_name=network_name, results_dir=results_dir))
        plt.close()


def save_results(dir_name,
                 trial_nums=[1, 2, 3, 4, 5],
                 network_names=['DNC', 'RNN-LN-FW'],
                 test_names=['QSUBJECT', 'QPOET', 'QEMCEE', 'QFRIEND', 'QDESSERT', 'QDRINK']):
    templates = {'RNN-LN-FW': 'test_analysis_results_{network_name}_30000epochs_trial{trial_num}_test_{test_name}.p_noise0_zerovectornoiseFalse',
            'DNC': 'test_analysis_results_{network_name}_30000epochs_trial{trial_num}_test_{test_name}.p_noise0_zerovectornoiseFalse'}

    results_df = pd.DataFrame(columns=['trial_num', 'test_name', 'accuracy', 'network_name'])

    for trial_num in trial_nums:
        for test_name in test_names:
            for network_name in network_names:
                with open(os.path.join(distributions_results_dir, dir_name, 'predictions', templates[network_name].format(network_name=network_name, trial_num=trial_num, test_name=test_name)), 'rb') as f:
                    a = pickle.load(f)
                try:
                    accuracy = sum(a['predictions'] == a['responses']) / (1.0 * len(a))
                except:
                    print(dir_name, trial_num, test_name, network_name)
                    accuracy = 0
                results_df = results_df.append({'trial_num': trial_num, 'test_name': test_name.lower().replace('', ''), 'accuracy': accuracy, 'network_name': network_name}, ignore_index=True)

    results_df.to_pickle('accuracies_{dir_name}.p'.format(dir_name=dir_name))


def plot_accuracies(dir_name,
        network_names=['DNC', 'RNN-LN-FW'],
        split_type='trial',
        chance_rate=1 / (96 + 30)):
    with open('accuracies_{dir_name}.p'.format(dir_name=dir_name), 'rb') as f:
        results_df = pickle.load(f)

    for network_name in network_names:
        results_df_network = results_df[results_df['network_name'] == network_name]
        for query_index, query in enumerate(queries):
            x = query_index + 1
            accuracies = np.array(results_df_network[results_df_network['test_name'] == query]['accuracy'].values)
            y_mean = np.mean(accuracies)
            y_min = np.min(accuracies)
            y_max = np.max(accuracies)
            y_err = [[y_mean - y_min], [y_max - y_mean]]
            plt.bar(x, y_mean, yerr=y_err, color=color_blue)
        query_titles = [query[:2].upper() + query[2:] for query in queries]
        plt.xticks(np.arange(1, len(queries) + 1), query_titles, fontsize=axis_fontsize)
        #plt.xlabel('Query')
        plt.ylabel('Test Accuracy', fontsize=axis_fontsize)
        plt.axhline(y=chance_rate, label="Chance Rate", color='k', linestyle='--')
        #plt.legend(loc='lower right', bbox_to_anchor=(1, 1))

        plt.savefig('figures/accuracies_{dir_name}_{network_name}_{split_type}'.format(dir_name=dir_name, network_name=network_name.replace("-", ""), split_type=split_type), dpi=600)
        plt.close()


if __name__ == '__main__':
    plot_bias_ttest(network_names=['DNC'], trial_nums=[1, 2, 3, 4, 5])
    plot_hists(network_names=['DNC'])
    #plot_bias_ttest(network_names=['RNN-LN-FW'], trial_nums=[1, 2, 3, 4, 5])
    #plot_hists(network_names=['RNN-LN-FW'])
    #plot_bias_ttest_diffs(network_names=['DNC'], trial_nums=[1, 2, 3, 4, 5])
    #plot_bias_ttest_diffs(network_names=['RNN-LN-FW'], trial_nums=[1, 2, 3, 4, 5])
    #dirs = ['variable_filler_distributions_A', 'variable_filler_distributions_B', 'variable_filler_distributions_all_randn_distribution', 'variable_filler_distributions_5050_AB']
    #for dir_name in dirs:
    #    plot_bias_ttest_diffs(dir_name, trial_nums=[1, 2, 3, 4, 5], network_names=['DNC'])
    #    save_results(dir_name, trial_nums=[1, 2, 3, 4, 5], network_names=['DNC'])
    #    plot_accuracies(dir_name, network_names=['DNC'])
    #    plot_bias_ttest_diffs(dir_name, trial_nums=[1, 2, 3, 4, 5], network_names=['RNN-LN-FW'])
    #    save_results(dir_name, trial_nums=[1, 2, 3, 4, 5], network_names=['RNN-LN-FW'])
    #    plot_accuracies(dir_name, network_names=['RNN-LN-FW'])
