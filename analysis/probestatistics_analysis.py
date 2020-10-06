import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import re

from scipy.stats import ttest_ind

def plot_test_results(test_results_dict, title, save_name):
    fig, ax = plt.subplots()
    trial_nums = list(test_results_dict.keys())
    x = range(len(trial_nums))
    values = list(test_results_dict.values())
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(trial_nums)
    ax.set_ylabel('test accuracy')
    ax.set_xlabel('trial num')
    ax.set_title(title)
    ax.set_ylim([0, 1])
    fig.tight_layout()
    plt.savefig(save_name)
    plt.close()

def t_test(odd_values, even_values):
    mean_odd = np.mean(odd_values)
    mean_even = np.mean(even_values)
    var_odd = np.var(odd_values)
    var_even = np.var(even_values)
    t_stat = (mean_odd - mean_even) / np.sqrt(var_odd/len(odd_values) + var_even/len(even_values))
    return t_stat

def compute_t_stats(logits_dict, f):
    statistics = []
    pvalues = []
    for trial_num, logits in logits_dict.items():
        f.write('trial {trial_num}\n'.format(trial_num=trial_num))
        #t_stat = t_test(logits[:,1::2], logits[:,::2])
        #f.write('t_stat {t_stat}\n'.format(t_stat=t_stat))
        statistic, pvalue = ttest_ind(logits[:,1::2].flatten(), logits[:,::2].flatten())
        statistics.append(statistic)
        pvalues.append(pvalue)
        f.write('scipy t_stat two-tailed {statistic}, pvalue {pvalue}\n'.format(statistic=statistic, pvalue=pvalue))
    return statistics, pvalues

def plot_mean_guesses(logits_dict, save_name, title):
    logits_values_list = list(logits_dict.values())
    even_values = [np.mean(logits_values[:,::2]) for logits_values in logits_values_list]
    odd_values = [np.mean(logits_values[:,1::2]) for logits_values in logits_values_list]
    trial_nums = list(logits_dict.keys())
    fig, ax = plt.subplots()
    x = np.arange(len(trial_nums))
    ax.set_xlabel('trial num')
    ax.set_ylabel('average predicted value')
    ax.set_title(title)
    ax.bar(x - 0.1, even_values, width=0.1, label='even', color='g')
    ax.bar(x + 0.1, odd_values, width=0.1, label='odd', color='r')
    ax.set_xticks(x)
    ax.set_xticklabels(trial_nums)
    ax.set_ylim([-0.5, 0.5])
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    plt.savefig(save_name, bbox_extra_artists=[legend], bbox_inches='tight')
    plt.close()

def parse_filename(filename, regex):
    info = re.match(regex, filename)
    if not info:
        return info
    return info.groupdict()

def get_max_epochs_dict(results_path, network_name, regex):
    filenames = os.listdir(results_path)
    network_filenames = [filename for filename in filenames if network_name+'_' in filename]
    trial_max_epochs_dict = dict()
    for filename in network_filenames:
        info = parse_filename(filename, regex)
        if not info:
            print(filename + ' not parsed')
            continue
        trial_num = int(info['trial_num'])
        num_epochs = int(info['num_epochs'])
        if trial_num not in trial_max_epochs_dict:
            trial_max_epochs_dict[trial_num] = num_epochs
        else:
            if num_epochs > trial_max_epochs_dict[trial_num]:
                trial_max_epochs_dict[trial_num] = num_epochs
    return trial_max_epochs_dict

def get_test_results(experiment_name, network_name, results_dir):
    results_path = os.path.join(results_dir, experiment_name, 'fixed_filler')
    regex = '(?P<network_name>\S+)_(?P<num_epochs>[0-9]+)epochs_trial(?P<trial_num>[0-9]+).p'
    trial_max_epochs_dict = get_max_epochs_dict(results_path, network_name, regex)
    test_results_dict = dict()
    for trial_num, num_epochs in trial_max_epochs_dict.items():
        filename = '{network_name}_results_{num_epochs}epochs_trial{trial_num}.p'.format(network_name=network_name, num_epochs=num_epochs, trial_num=trial_num)
        with open(os.path.join(results_path, filename), 'rb') as f:
            file_results = pickle.load(f)
            test_results_dict[trial_num] = file_results['test_accuracy'][-1]
    return test_results_dict

def get_unseen_test_results(experiment_name, network_name, results_dir, test_name):
    results_path = os.path.join(results_dir, experiment_name, 'fixed_filler')
    regex = '(?P<network_name>\S+)_(?P<num_epochs>[0-9]+)epochs_trial(?P<trial_num>[0-9]+).p'
    trial_max_epochs_dict = get_max_epochs_dict(results_path, network_name, regex)
    test_results_dict = dict()
    for trial_num, num_epochs in trial_max_epochs_dict.items():
        filename = '{network_name}_results_{num_epochs}epochs_trial{trial_num}_tests.p'.format(network_name=network_name, num_epochs=num_epochs, trial_num=trial_num)
        with open(os.path.join(results_path, filename), 'rb') as f:
            file_results = pickle.load(f)
            test_results_dict[trial_num] = file_results['accuracies'][test_name][-1]
    return test_results_dict

def get_logits(experiment_name, network_name, probe_name, results_dir, settozero):
    results_path = os.path.join(results_dir, experiment_name, 'fixed_filler', 'probe_statistics_'+probe_name+'_settozero{settozero}'.format(settozero=settozero))
    if not os.path.exists(results_path):
        return
    regex = 'logits_(?P<network_name>\S+)_(?P<num_epochs>[0-9]+)epochs_trial(?P<trial_num>[0-9]+)_(?P<probe_name>\S+).p'
    trial_max_epochs_dict = get_max_epochs_dict(results_path, network_name, regex)
    logits_dict = dict()
    for trial_num, num_epochs in trial_max_epochs_dict.items():
        filename = 'logits_{network_name}_{num_epochs}epochs_trial{trial_num}_{probe_name}.p_settozero{settozero}'.format(experiment_name=experiment_name, probe_name=probe_name, network_name=network_name, num_epochs=num_epochs, trial_num=trial_num, settozero=settozero)
        print(filename)
        with open(os.path.join(results_path, filename), 'rb') as f:
            logits_dict[trial_num] = pickle.load(f)[0]
    return logits_dict

def main(args):
    network_names = args.network_names.split('_')
    experiment_names = args.experiment_names.split('-')
    probe_names = args.probe_names.split('-')
    test_names = args.test_names.split('-')
    f = open(os.path.join(args.plot_dir, 't_stats'), 'w')
    t_stats_dict = {}
    for network_name in network_names:
        t_stats_dict[network_name] = {}
        for experiment_name in experiment_names:
            t_stats_dict[network_name][experiment_name] = {}
            percentage = re.match('probestatisticsretention_percentageindistribution(?P<percentage>\S+)_normalizefillerdistributionFalse', experiment_name).group(1)
            test_results_dict = get_test_results(experiment_name, network_name, args.results_dir)
            plot_test_results(test_results_dict, title='Test Accuracies, {network_name}\n{percentage}%'.format(network_name=network_name, percentage=percentage), save_name=os.path.join(args.plot_dir, 'accuracies', experiment_name + '_' + network_name))
            for test_name in test_names:
                unseen_test_results_dict = get_unseen_test_results(experiment_name, network_name, args.results_dir, test_name)
                plot_test_results(unseen_test_results_dict, title='Test Accuracies, {network_name}\n{percentage}%\n{test_name}'.format(test_name=test_name, network_name=network_name, percentage=percentage), save_name=os.path.join(args.plot_dir, 'accuracies', experiment_name + '_' + network_name + '_' + test_name))
            for probe_name in probe_names:
                t_stats_dict[network_name][experiment_name][probe_name] = {}
                for settozero in ['True', 'False']:
                    logits_dict = get_logits(experiment_name, network_name, probe_name, args.results_dir, settozero)
                    with open(os.path.join(args.plot_dir, 'logits_{probe_name}_{experiment_name}_{network_name}_settozero{settozero}.p'.format(probe_name=probe_name, experiment_name=experiment_name, network_name=network_name, settozero=settozero)), 'wb') as f2:
                        pickle.dump(logits_dict, f2)
                    if not logits_dict:
                        continue
                    f.write('experiment name: {experiment_name}\n'.format(experiment_name=experiment_name))
                    f.write('network name: {network_name}\n'.format(network_name=network_name))
                    f.write('probe name: {probe_name}\n'.format(probe_name=probe_name))
                    f.write('set to zero: {settozero}\n'.format(settozero=settozero))
                    statistics, pvalues = compute_t_stats(logits_dict, f)
                    t_stats_dict[network_name][experiment_name][probe_name]['settozero{settozero}'.format(settozero=settozero)] = {
                            'statistics':statistics,
                            'pvalues':pvalues
                            }
                    plot_mean_guesses(logits_dict, save_name=os.path.join(args.plot_dir, 'logits', experiment_name + '_' + network_name + '_' + probe_name + '_settozero' + settozero), title='Mean Predictions\n{percentage}% {network_name}'.format(percentage=percentage, network_name=network_name))
    with open(os.path.join(args.plot_dir, 't_stats_dict.p'), 'wb') as f:
        pickle.dump(t_stats_dict, f)
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='/home/cc27/Thesis/generalized_schema_learning/results/')
    parser.add_argument('--plot_dir', type=str, default='/home/cc27/Thesis/generalized_schema_learning/probe_statistics_plots')
    parser.add_argument('--network_names', type=str, help='network names, separated by _', default='RNN-LN_LSTM-LN_RNN-LN-FW_NTM2')
    parser.add_argument('--experiment_names', type=str, help='experiment names, separated by -', default='probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse-probestatisticsretention_percentageindistribution25_normalizefillerdistributionFalse-probestatisticsretention_percentageindistribution50_normalizefillerdistributionFalse-probestatisticsretention_percentageindistribution75_normalizefillerdistributionFalse-probestatisticsretention_percentageindistribution100_normalizefillerdistributionFalse')
    parser.add_argument('--probe_names', type=str, help='probe names, separated by -', default='test_QEmcee_replaceemcee-test_QFriend_replacefriend-test_QPoet_replacepoet-test_QSubject_replacesubject')
    parser.add_argument('--test_names', type=str, help='test dataset names, separated by -', default='test_QEmcee_unseen_middistribution-test_QSubject_unseen_indistribution-test_QFriend_unseen_indistribution-test_QSubject_unseen_middistribution-test_QEmcee_unseen_outofdistribution-test_QEmcee_unseen_indistribution-test_QPoet_unseen_middistribution-test_QPoet_unseen_outofdistribution-test_QPoet_unseen_indistribution-test_QFriend_unseen_middistribution-test_QSubject_unseen_outofdistribution-test_QFriend_unseen_outofdistribution')
    args = parser.parse_args()

    if not os.path.exists(args.plot_dir):
        os.makedirs(os.path.join(args.plot_dir, 'logits'))
        os.makedirs(os.path.join(args.plot_dir, 'accuracies'))

    main(args)
