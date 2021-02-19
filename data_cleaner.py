import os
import re

def clean_checkpoints(checkpoint_dir):
    experiment_checkpoint_dirs = os.listdir(checkpoint_dir)
    for experiment_checkpoint_dir in experiment_checkpoint_dirs:
        experiment_dir = os.path.join(checkpoint_dir, experiment_checkpoint_dir)
        for filler_dir in os.listdir(experiment_dir):
            filler_dir_full = os.path.join(experiment_dir, filler_dir)
            for network_dir in os.listdir(filler_dir_full):
                network_dir_full = os.path.join(filler_dir_full, network_dir)
                for trial_dir in os.listdir(network_dir_full):
                    trial_dir_full = os.path.join(network_dir_full, trial_dir)
                    filenames = os.listdir(trial_dir_full)
                    max_epoch_num = 0
                    for filename in filenames:
                        if filename == 'checkpoint':
                            continue
                        epoch_num = int(filename.split('.')[0].split('-')[-1])
                        if epoch_num > max_epoch_num:
                            max_epoch_num = epoch_num
                    filenames_to_remove = [filename for filename in filenames if str(max_epoch_num) not in filename and filename != 'checkpoint']
                    if len(filenames_to_remove) > 0:
                        print('removing: ', filenames_to_remove)
                        print('keeping: ', set(filenames) - set(filenames_to_remove))
                        do_delete = raw_input('Delete these files? (y/n)')
                        if do_delete == 'y':
                            for filename in filenames_to_remove:
                                os.remove(os.path.join(trial_dir_full, filename))


def clean_results(results_dir):
    experiment_results_dirs = os.listdir(results_dir)
    for experiment_results_dir in ['variablefiller_AllQs', 'fixedfiller_AllQs', 'generate_train3roles_testnewrole_withunseentestfillers_1000personspercategory_24000train_120test', 'generate_train3roles_testnewrole_withunseentestfillers_1000personspercategory_24000train_120test_shuffled']: #experiment_results_dirs:
        experiment_dir = os.path.join(results_dir, experiment_results_dir)
        for filler_dir in os.listdir(experiment_dir):
            filler_dir_full = os.path.join(experiment_dir, filler_dir)
            print('*****************%s****************' % filler_dir_full)
            for filler_dir_2 in os.listdir(filler_dir_full):
                filler_dir_full_2 = os.path.join(filler_dir_full, filler_dir_2)
                filenames = os.listdir(filler_dir_full_2)
                for network_name in ['RNN-LN-FW', 'NTM2-xl', 'RNN-LN_', 'LSTM-LN_', 'LSTM-LN-five']:
                    max_epochs_dict = {}
                    network_filenames = [filename for filename in filenames if network_name in filename]
                    for filename in network_filenames:
                        filepath = os.path.join(filler_dir_full_2, filename)
                        epoch_num = int(filename.split('epochs')[0].split('_')[-1])
                        trial_num = int(re.match(r".*?trial?(?P<trial_num>\d+).*\.p", filename).group('trial_num'))
                        if trial_num in max_epochs_dict:
                            max_epochs_dict[trial_num]['filenames'].append(filepath)
                            if epoch_num > max_epochs_dict[trial_num]['max_epochs']:
                                max_epochs_dict[trial_num]['max_epochs'] = epoch_num
                        else:
                            max_epochs_dict[trial_num] = {'max_epochs': epoch_num, 'filenames': [filepath]}
                    for trial_num in max_epochs_dict:
                        filenames_to_remove = [filename for filename in max_epochs_dict[trial_num]['filenames'] if str(max_epochs_dict[trial_num]['max_epochs']) + 'epochs' not in filename]
                        filenames_to_keep = set(max_epochs_dict[trial_num]['filenames']) - set(filenames_to_remove)
                        if len(filenames_to_remove) > 0:
                            print('removing: ', filenames_to_remove)
                            print('keeping: ', filenames_to_keep)
                            do_delete = raw_input('Delete these files? (y/n)')
                            if do_delete == 'y':
                                for filename in filenames_to_remove:
                                    os.remove(os.path.join(filler_dir_full_2, filename))

if __name__ == '__main__':
    #clean_checkpoints('/home/cc27/Thesis/generalized_schema_learning/checkpoints')
    clean_results('/home/cc27/Thesis/generalized_schema_learning/results')
