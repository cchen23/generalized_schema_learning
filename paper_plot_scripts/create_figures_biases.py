'''Fig 9, 10, 11.'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import plotnine as p9
import os
import scipy


def save_results(dir_name,
                 trial_nums=range(4, 11),
                 network_names=['NTM2', 'RNN-LN-FW'],
                 test_names=['QSUBJECT', 'QPOET', 'QEMCEE_double', 'QFRIEND', 'QDESSERT_double', 'QDRINK']):
    template = 'test_analysis_results_{network_name}_30000epochs_trial{trial_num}_test_{test_name}.p'

    results_df = pd.DataFrame(columns=['trial_num', 'test_name', 'accuracy', 'network_name'])

    for trial_num in trial_nums:
        for test_name in test_names:
            for network_name in network_names:
                with open(os.path.join(dir_name, 'predictions', template.format(network_name=network_name, trial_num=trial_num, test_name=test_name)), 'rb') as f:
                    a = pickle.load(f)

                accuracy = sum(a['predictions'] == a['responses']) / len(a)
                results_df = results_df.append({'trial_num': trial_num, 'test_name': test_name.lower().replace('_double', ''), 'accuracy': accuracy, 'network_name': network_name}, ignore_index=True)

    results_df.to_pickle(f'accuracies_{dir_name}.p')


def plot_accuracies(dir_name,
        network_names=['NTM2', 'RNN-LN-FW']):
    with open(f'accuracies_{dir_name}.p', 'rb') as f:
        results_df = pickle.load(f)

    for network_name in network_names:
        results_df_network = results_df[results_df['network_name'] == network_name]
        min_trial_num = np.min(results_df_network['trial_num'])
        results_df_network['trial_num'] = results_df_network['trial_num'] - min_trial_num
        plot = p9.ggplot(results_df_network, p9.aes(x='test_name', y='accuracy')) +\
                p9.geom_col(stat=p9.stats.stat_summary(fun_y=np.mean), width=0.5) +\
                p9.scales.ylim(0, 1) +\
                p9.labels.labs(x='Query',
                      y='Test Accuracy',
                      title=f'{network_name} Accuracy on Fillers from {dir_name.split("_")[-1].replace("distribution", "C")}\n(Mean over {len(np.unique(results_df_network["trial_num"]))} trials)') +\
                p9.scale_color_discrete(guide=False)

        plot.save(f'plots/accuracies_{dir_name}_{network_name.replace("-", "")}')


def plot_bias_ttest(network_names=['NTM2', 'RNN-LN-FW'],
        trial_nums=range(1, 3)):
    results_df = pd.DataFrame(columns=['network_name', 'trial_num', 'ttest_stat', 'pvalue', 'query'])
    queries = ['qdessert', 'qemcee', 'qpoet', 'qsubject', 'qfriend', 'qdrink']
    with open('../predictions/wordslist.p', 'rb') as f:
        wordlist = pickle.load(f)
    for trial_num in trial_nums:
        for network_name in network_names:
            with open(f'./variable_filler_distributions/predictions/test_analysis_results_{network_name}_30000epochs_trial{trial_num}_test.p', 'rb') as f:
                r = pickle.load(f)
            logits = np.load(f'./variable_filler_distributions/predictions/logits_{network_name}_30000epochs_trial{trial_num}_test.p.npz')['arr_0']

            indices_by_query = {}

            for idx, row in r.iterrows():
                query = wordlist[row['inputs'][-1]]
                if query not in indices_by_query:
                    indices_by_query[query] = []
                indices_by_query[query].append(idx)

            for query, indices in indices_by_query.items():
                query_logits = logits[indices]
                s, pvalue = scipy.stats.ttest_ind(query_logits[:, ::2].ravel(), query_logits[:, 1::2].ravel())
                results_df = results_df.append({'network_name': network_name, 'trial_num': trial_num, 'ttest_stat': s, 'pvalue': pvalue, 'query': query.lower(), 'even_logits': query_logits[:, ::2].ravel(), 'odd_logits': query_logits[:, 1::2].ravel()}, ignore_index=True)

    query_cat = pd.Categorical(results_df['query'], categories=queries)
    results_df = results_df.assign(query_cat=query_cat)
    results_df.to_pickle('logit_pvalues.p')
    for network_name in network_names:
        results_df_network = results_df[results_df['network_name'] == network_name]
        min_trial_num = np.min(results_df_network['trial_num'])
        results_df_network['trial_num'] = results_df_network['trial_num'] - min_trial_num
        plot = p9.ggplot(results_df_network, p9.aes(x='query_cat', y='ttest_stat', fill='trial_num')) +\
                p9.geom_col(stat='identity', position='dodge') +\
                p9.labels.labs(x='Query',
                      y='ttest Statistic',
                      title=f'{network_name} t-Test between Even and Odd Indices on Ambiguous Examples')

        plot.save(f'plots/ttest_{network_name}')


def plot_hists(network_names=['NTM2', 'RNN-LN-FW']):
    with open('logit_pvalues.p', 'rb') as f:
        results_df = pickle.load(f)

    bins = np.arange(-0.2, 0.2, 0.01)
    queries = ['qdessert', 'qemcee', 'qpoet', 'qsubject', 'qfriend', 'qdrink']
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
            axes[idx].hist(query_odd_logits, label='odd', bins=bins)
            axes[idx].hist(query_even_logits, label='even', bins=bins)
            axes[idx].set_ylabel(query.lower(), fontsize=8)
        axes[0].legend()
        axes[0].set_title(f'{network_name} Histogram of Predicted Values on Ambiguous Inputs', fontsize=12)
        plt.savefig(f'plots/hist_{network_name}')


def plot_bias_ttest_diffs(results_dir,
        network_names=['NTM2', 'RNN-LN-FW'],
        trial_nums=range(1, 6)):
    results_df = pd.DataFrame(columns=['network_name', 'trial_num', 'ttest_stat', 'pvalue', 'query'])
    query_to_test_name = {'qdessert': 'QDESSERT_double',
            'qemcee': 'QEMCEE_double',
            'qpoet': 'QPOET',
            'qsubject': 'QSUBJECT',
            'qfriend': 'QFRIEND',
            'qdrink': 'QDRINK'}
    for trial_num in trial_nums:
        for network_name in network_names:
            for query in query_to_test_name:
                logits = np.load(f'./{results_dir}/predictions/logits_{network_name}_30000epochs_trial{trial_num}_test_{query_to_test_name[query]}.p.npz')
                logits_pred = logits['predicted_logits']
                logits_true = logits['true_logits']
                logits_diff = logits_pred - logits_true

                s, pvalue = scipy.stats.ttest_ind(logits_diff[:, ::2].ravel(), logits_diff[:, 1::2].ravel())
                results_df = results_df.append({'network_name': network_name, 'trial_num': trial_num, 'ttest_stat': s, 'pvalue': pvalue, 'query': query, 'even_logits': logits_diff[:, ::2].ravel(), 'odd_logits': logits_diff[:, 1::2].ravel()}, ignore_index=True)

    query_cat = pd.Categorical(results_df['query'], categories=query_to_test_name.keys())
    results_df = results_df.assign(query_cat=query_cat)
    results_df.to_pickle('logit_pvalues.p')
    for network_name in network_names:
        results_df_network = results_df[results_df['network_name'] == network_name]
        min_trial_num = np.min(results_df_network['trial_num'])
        results_df_network['trial_num'] = results_df_network['trial_num'] - min_trial_num
        plot = p9.ggplot(results_df_network, p9.aes(x='query_cat', y='ttest_stat', fill='trial_num')) +\
                p9.geom_col(stat='identity', position='dodge') +\
                p9.labels.labs(x='Query',
                      y='ttest Statistic',
                      title=f'{network_name} t-Test between Even and Odd Indices on {results_dir} Examples')

        plot.save(f'plots/ttest_{network_name}_{results_dir}')


if __name__ == '__main__':
    #dirs = ['variable_filler_distributions_A', 'variable_filler_distributions_B', 'variable_filler_distributions_all_randn_distribution', 'variable_filler_distributions_5050_AB']
    dirs = ['variable_filler_distributions_A', 'variable_filler_distributions_B', 'variable_filler_distributions_all_randn_distribution']
    for dir_name in dirs:
        plot_bias_ttest_diffs(dir_name)
        #save_results(dir_name, trial_nums=range(1, 6), network_names=['RNN-LN-FW', 'NTM2'])
        #plot_accuracies(dir_name, network_names=['RNN-LN-FW', 'NTM2'])
    #plot_bias_ttest(['RNN-LN-FW', 'NTM2'], range(1, 6))
    #plot_hists()
