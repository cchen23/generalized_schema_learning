"""Modules to remove old checkpoints and results."""
import directories
import fnmatch
import os
import sys

base_dir = directories.base_dir

def get_fileepochs(file):
    """Parse epoch number from filename.

    Assumes filename convention in train.py.
    """
    return int(file.split('epochs')[0].split('_')[-1])

def clean_checkpoints(experiment_foldername):
    """Remove all checkpoint files except those with the most train epochs.

    Assumes directory naming convention in train.py.

    Args:
        experiment_foldername: Name of directory to clean. Path from folder
                               directly inside checkpoints/ directory through
                               folder containing architecture folders.
    """
    print("Cleaning checkpoints for %s" % experiment_foldername)
    experiment_directory = directories.base_dir + "checkpoints/" + experiment_foldername + "/"
    print(experiment_directory)
    walker_result = next(os.walk(experiment_directory))
    subdirs = walker_result[1]
    for subdir in subdirs:
        print(subdir)
        architecture_directory = experiment_directory + subdir + "/"
        walker_result = next(os.walk(architecture_directory))
        subsubdirs = walker_result[1]
        for subsubdir in subsubdirs:
            checkpoint_nums = set()
            dropout_directory = architecture_directory + subsubdir + "/"
            dropout_directory_files = next(os.walk(dropout_directory))[2]
            # Get checkpoint nums.
            for filename in dropout_directory_files:
                if filename != "checkpoint" and "index" in filename:
                    checkpoint_num = int(filename.split('.')[0].split('-')[-1])
                    checkpoint_nums.add(checkpoint_num)
            # Remove all except last checkpoint num.
            savedcheckpoint_num = max(checkpoint_nums)
            print(checkpoint_nums)
            for filename in dropout_directory_files:
                if str(savedcheckpoint_num) in filename or filename == "checkpoint":
                    print("skipping %s" % filename)
                    continue
                file_path = dropout_directory + filename
                print("removing %s" % (file_path))
                os.remove(file_path)


def clean_results(experiment_foldername):
    """Remove all results files except those with the most train epochs.

    Assumes directory creation method used in train.py.

    Args:
        experiment_foldername: Name of directory to clean. Path from folder
                               directly inside results/ directory through
                               folder containing architecture folders.
    """
    print("Cleaning results for %s" % experiment_foldername)
    RESULTSFILE_TEMPLATES = ['%s_results_%depochs_10dropout_split.p', '%s_results_%depochs_10dropout.p']
    MATCH_TEMPLATES = ['%s_results_*epochs_10dropout_split.p', '%s_results_*epochs_10dropout.p']
    for MATCH_TEMPLATE, RESULTSFILE_TEMPLATE in zip(MATCH_TEMPLATES, RESULTSFILE_TEMPLATES):
        experiment_directory = directories.base_dir + "results/" + experiment_foldername
        print(experiment_directory)
        maxepochs = {
            'CONTROL': 0,
            'DNC': 0,
            'RNN-LN-FW': 0,
            'GRU-LN': 0,
            'LSTM-LN': 0,
            'NTM2': 0,
            'RNN-LN': 0
        }
        architectures = ['CONTROL', 'DNC', 'RNN-LN-FW', 'GRU-LN', 'LSTM-LN', 'NTM2', 'RNN-LN']
        removed_files = []
        for file in os.listdir(experiment_directory):
            print("file: ", file)
            if file in removed_files:
                continue
            for architecture in architectures:
                if fnmatch.fnmatch(file, MATCH_TEMPLATE % architecture):
                    fileepochs = get_fileepochs(file)
                    if fileepochs < maxepochs[architecture]:
                        print(file)
                        os.remove(experiment_directory + file)
                        removed_files.append(file)
                    else:
                        if maxepochs[architecture] > 0:
                            print(RESULTSFILE_TEMPLATE % (architecture, maxepochs[architecture]))
                            os.remove(experiment_directory + RESULTSFILE_TEMPLATE % (architecture, maxepochs[architecture]))
                            removed_files.append(file)
                        maxepochs[architecture] = fileepochs



if __name__ == '__main__':
    clean_option = sys.argv[1]

    if clean_option == "checkpoints":
        experiment_foldernames = sys.argv[2:]
        for experiment_foldername in experiment_foldernames:
            clean_checkpoints(experiment_foldername)

    elif clean_option == "results":
        experiment_foldernames = sys.argv[2:]
        for experiment_foldername in experiment_foldernames:
            clean_results(experiment_foldername)
