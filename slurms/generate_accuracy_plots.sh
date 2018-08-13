#!/usr/bin/env bash
#SBATCH -J 'generate_accuracy_plots'
#SBATCH -o outputs/slurm-%j-generate_accuracy_plots.out
#SBATCH -p all
#SBATCH -t 1000
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:1

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "
echo "With access to cpu id(s): "
cat /proc/$$/status | grep Cpus_allowed_list
echo "Array Allocation Number: $SLURM_ARRAY_JOB_ID"
echo "Array Index: $SLURM_ARRAY_TASK_ID"

python /home/cc27/Thesis/generalized_schema_learning/analysis/generate_accuracy_plots.py --exp_folder=fixedfiller_gensymbolicstates_100000_1_AllQs/fixed_filler/ --exp_title="Fixed Random Representation\nAllQs, Previously Seen Fillers" --trial_num=0
python /home/cc27/Thesis/generalized_schema_learning/analysis/generate_accuracy_plots.py --exp_folder=fixedfiller_gensymbolicstates_100000_1_testunseen_AllQs/fixed_filler/ --exp_title="Fixed Random Representation\nAllQs, Previously Unseen Fillers" --trial_num=0
python /home/cc27/Thesis/generalized_schema_learning/analysis/generate_accuracy_plots.py --exp_folder=variablefiller_gensymbolicstates_100000_1_testunseen_AllQs/variable_filler/ --exp_title="Variable Random Representation\nAllQs, Previously Unseen Fillers" --trial_num=0
python /home/cc27/Thesis/generalized_schema_learning/analysis/generate_accuracy_plots.py --exp_folder=variablefiller_gensymbolicstates_100000_1_testunseen_QSubject/variable_filler/ --exp_title="Variable Random Representation\nQSubject, Previously Unseen Fillers" --trial_num=0
