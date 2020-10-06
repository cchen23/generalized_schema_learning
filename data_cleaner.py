import os

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
    for experiment_results_dir in experiment_results_dirs:
        experiment_dir = os.path.join(results_dir, experiment_results_dir)
        for filler_dir in os.listdir(experiment_dir):
            filler_dir_full = os.path.join(experiment_dir, filler_dir)
            print('*****************%s****************' % filler_dir_full)
            filenames = os.listdir(filler_dir_full)
            for network_name in ['RNN-LN-FW', 'NTM2-xl', 'RNN-LN_', 'LSTM-LN']:
                max_epoch_num = 0
                network_filenames = [filename for filename in filenames if network_name in filename]
                for filename in network_filenames:
                    filepath = os.path.join(filler_dir_full, filename)
                    epoch_num = int(filename.split('epochs')[0].split('_')[-1])
                    if epoch_num > max_epoch_num:
                        max_epoch_num = epoch_num
                filenames_to_remove = [filename for filename in network_filenames if str(max_epoch_num) not in filename]
                filenames_to_keep = set(network_filenames) - set(filenames_to_remove)
                if len(filenames_to_remove) > 0:
                    print('removing: ', filenames_to_remove)
                    print('keeping: ', filenames_to_keep)
                    do_delete = raw_input('Delete these files? (y/n)')
                    if do_delete == 'y':
                        for filename in filenames_to_remove:
                            os.remove(os.path.join(filler_dir_full, filename))

if __name__ == '__main__':
    clean_checkpoints('/home/cc27/Thesis/generalized_schema_learning/checkpoints')
    clean_results('/home/cc27/Thesis/generalized_schema_learning/results')
