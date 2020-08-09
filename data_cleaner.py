import os

def clean_checkpoints(checkpoint_dir):
    experiment_checkpoint_dirs = os.listdir(checkpoint_dir)
    for experiment_checkpoint_dir in experiment_checkpoint_dirs:
        current_dir = os.path.join(checkpoint_dir, experiment_checkpoint_dir)
        for filler_dir in os.listdir(current_dir):
            current_dir = os.path.join(current_dir, filler_dir)
            for network_dir in os.listdir(current_dir):
                current_dir = os.path.join(current_dir, network_dir)
                for trial_dir in os.listdir(current_dir):
                    current_dir = os.path.join(current_dir, trial_dir)
                    filenames = os.listdir(current_dir)
                    max_num_epochs = 0
                    for filename in filenames:
                        if filename == 'checkpoint':
                            continue
                        epoch_num = int(filename.split('-')[1].split('.')[0])
                        if epoch_num > max_epoch_num:
                            max_epoch_num = epoch_num
                    filenames_to_remove = [filename for filename in filenames if max_epoch_num not in filename]
                    print('removing: ', filenames_to_remove)
                    print('keeping: ', set(filenames) - set(filenames_to_remove))
                    do_delete = input('Delete these files? (y/n)')
                    if do_delete == 'y':
                        for filename in filenames_to_remove:
                            os.remove(os.path.join(current_dir, filename))


def clean_results(results_dir):
    experiment_results_dirs = os.listdir(results_dir)
    for experiment_results_dir in experiment_results_dirs:
        current_dir = os.path.join(results_dir, experiment_results_dir)
        for filler_dir in os.listdir(current_dir):
            current_dir = os.path.join(current_dir, filler_dir)
            filenames = os.listdir(current_dir)
            max_num_epochs = 0
            for filename in filenames:
                filepath = os.path.join(current_dir, filename)
                if os.path.isdir(filepath):
                    print(filepath)
                    do_delete = input('Delete this directory? (y/n)')
                    if do_delete == 'y':
                        os.rmdir(filepath)
                else:
                    epoch_num = int(filename.split('trial')[1].split('_')[0])
                    if epoch_num > max_epoch_num:
                        max_epoch_num = epoch_num
            filenames_to_remove = [filename for filename in filenames if max_epoch_num not in filename]
            print('removing: ', filenames_to_remove)
            print('keeping: ', set(filenames) - set(filenames_to_remove))
            do_delete = input('Delete these files? (y/n)')
            if do_delete == 'y':
                for filename in filenames_to_remove:
                    os.remove(os.path.join(current_dir, filename))
