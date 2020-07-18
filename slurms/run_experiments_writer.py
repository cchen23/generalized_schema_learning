import sys
sys.path.append("../")
from directories import base_dir

def write_run_training(experiment_name, filler_type, architecture_name, num_epochs, trial_num, regime=None):
    experiment_filename = experiment_name.replace("_","")
    architecture_filename = architecture_name.replace("-","")
    fillertype_filename = filler_type.replace("_","")
    f = open("run_train_%s.sh" % architecture_filename, "w")
    f.write("#!/usr/bin/env bash\n")
    f.write("#SBATCH -J 'runtrain_%s_%s_%s_trial%d'\n" % (experiment_filename, architecture_filename, fillertype_filename, trial_num))
    f.write("#SBATCH -o outputs/slurm-%%j-runtrain_%s_%s_%s_trial%d.out\n" % (experiment_filename, architecture_filename, fillertype_filename, trial_num))
    f.write("#SBATCH -p all\n")
    f.write("#SBATCH -t 8640\n")
    f.write("#SBATCH -N 1\n")
    f.write("#SBATCH --ntasks-per-node=4\n")
    f.write("#SBATCH --ntasks-per-socket=2\n")
    f.write("#SBATCH --gres=gpu:1\n")
    f.write("\n")
    f.write("echo \"In the directory: `pwd` \"\n")
    f.write("echo \"As the user: `whoami` \"\n")
    f.write("echo \"on host: `hostname` \"\n")
    f.write("echo \"With access to cpu id(s): \"\n")
    f.write("cat /proc/$$/status | grep Cpus_allowed_list\n")
    f.write("echo \"Array Allocation Number: $SLURM_ARRAY_JOB_ID\"\n")
    f.write("echo \"Array Index: $SLURM_ARRAY_TASK_ID\"\n")
    f.write("\n")
    f.write("echo \"Run CSW %s experiment with %s\"\n" % (experiment_name, architecture_name))
    f.write("module load anaconda/4.4.0\n")
    f.write("source activate thesis\n")
    f.write("module load cudnn/cuda-9.0/7.0.3\n")
    f.write("module load cudatoolkit/9.0\n")
    f.write("date\n")
    f.write("echo \"***************************************** Running CSW %s Training with %s *****************************************\"\n" % (experiment_name, architecture_name))
    f.write("python %srun_experiment.py --function=train --exp_name=%s --filler_type=%s --model_name=%s --num_epochs=%d --trial_num=%d\n" % (base_dir, experiment_name, filler_type, architecture_name, num_epochs, trial_num))
    if "curriculum" in experiment_name:
        f.write("--regime=%s " % regime)
    f.write("date\n")
    f.write("echo \"Finished\"\n")
    f.close()

def write_run_test(experiment_name, filler_type, test_filename, trial_num):
    experiment_filename = experiment_name.replace("_","")
    fillertype_filename = filler_type.replace("_","")
    f = open("run_test_all.sh", "w")
    f.write("#!/usr/bin/env bash\n")
    f.write("#SBATCH -J 'runtest_%s_%s_%s_trial%d'\n" % (experiment_filename, fillertype_filename, test_filename[:-2], trial_num))
    f.write("#SBATCH -o outputs/slurm-%%j-runexperiments_%s_%s_%s_trial%d.out\n" % (experiment_filename, fillertype_filename, test_filename[:-2], trial_num))
    f.write("#SBATCH -p all\n")
    f.write("#SBATCH -t 1000\n")
    f.write("#SBATCH -N 1\n")
    f.write("#SBATCH --ntasks-per-node=4\n")
    f.write("#SBATCH --ntasks-per-socket=2\n")
    f.write("#SBATCH --gres=gpu:1\n")
    f.write("\n")
    f.write("echo \"In the directory: `pwd` \"\n")
    f.write("echo \"As the user: `whoami` \"\n")
    f.write("echo \"on host: `hostname` \"\n")
    f.write("echo \"With access to cpu id(s): \"\n")
    f.write("cat /proc/$$/status | grep Cpus_allowed_list\n")
    f.write("echo \"Array Allocation Number: $SLURM_ARRAY_JOB_ID\"\n")
    f.write("echo \"Array Index: $SLURM_ARRAY_TASK_ID\"\n")
    f.write("\n")
    f.write("echo \"Run CSW %s test with all architectures\"\n" % (experiment_name))
    f.write("module load anaconda/4.4.0\n")
    f.write("source activate thesis\n")
    f.write("module load cudnn/cuda-9.0/7.0.3\n")
    f.write("module load cudatoolkit/9.0\n")
    architectures = ["RNN-LN", "RNN-LN-FW", "LSTM-LN", "NTM2"]
    for architecture_name in architectures:
        f.write("echo \"***************************************** Running CSW %s Testing with %s *****************************************\"\n" % (experiment_name, architecture_name))
        f.write("python %srun_experiment.py --function=test --exp_name=%s --filler_type=%s --model_name=%s --test_filename=%s --trial_num=%d\n" % (base_dir, experiment_name, filler_type, architecture_name, test_filename, trial_num))
    f.write("echo \"Finished\"\n")
    f.close()

def write_run_analysis(experiment_name, filler_type, test_filename, trial_num):
    experiment_filename = experiment_name.replace("_","")
    fillertype_filename = filler_type.replace("_","")
    f = open("run_analysis.sh", "w")
    f.write("#!/usr/bin/env bash\n")
    f.write("#SBATCH -J 'runanalysis_%s_%s_%s_trial%d'\n" % (experiment_filename, fillertype_filename, test_filename[:-2], trial_num))
    f.write("#SBATCH -o outputs/slurm-%%j-runanalysis_%s_%s_%s_trial%d.out\n" % (experiment_filename, fillertype_filename, test_filename[:-2], trial_num))
    f.write("#SBATCH -p all\n")
    f.write("#SBATCH -t 500\n")
    f.write("#SBATCH -N 1\n")
    f.write("#SBATCH --ntasks-per-node=4\n")
    f.write("#SBATCH --ntasks-per-socket=2\n")
    f.write("#SBATCH --gres=gpu:1\n")
    f.write("\n")
    f.write("echo \"In the directory: `pwd` \"\n")
    f.write("echo \"As the user: `whoami` \"\n")
    f.write("echo \"on host: `hostname` \"\n")
    f.write("echo \"With access to cpu id(s): \"\n")
    f.write("cat /proc/$$/status | grep Cpus_allowed_list\n")
    f.write("echo \"Array Allocation Number: $SLURM_ARRAY_JOB_ID\"\n")
    f.write("echo \"Array Index: $SLURM_ARRAY_TASK_ID\"\n")
    f.write("\n")
    f.write("echo \"Run CSW %s analysis with all architectures\"\n" % (experiment_name))
    f.write("module load anaconda/4.4.0\n")
    f.write("source activate thesis\n")
    f.write("module load cudnn/cuda-9.0/7.0.3\n")
    f.write("module load cudatoolkit/9.0\n")
    architectures = ["NTM2"]
    for architecture_name in architectures:
        for trial_num in range(1):
            f.write("echo \"***************************************** Running CSW %s Testing with %s *****************************************\"\n" % (experiment_name, architecture_name))
            f.write("python %srun_experiment.py --function=analyze --exp_name=%s --filler_type=%s --model_name=%s --test_filename=%s --trial_num=%d\n" % (base_dir, experiment_name, filler_type, architecture_name, test_filename, trial_num))
    f.write("echo \"Finished\"\n")
    f.close()

if __name__ == '__main__':
    experiment_name = "variablefiller_gensymbolicstates_100000_1_testunseen_AllQs"
    filler_type = "variable_filler"
    regime = None
    test_filename = "test_analyze.p"
    num_epochs = 7500
    architectures = ["NTM2"]
    trial_num = 1
    # write_run_test(experiment_name, filler_type, test_filename, trial_num)
    # write_run_analysis(experiment_name, filler_type, test_filename, trial_num)
    for architecture_name in architectures:
        write_run_training(experiment_name, filler_type, architecture_name, num_epochs, trial_num, regime)
