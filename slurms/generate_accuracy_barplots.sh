#!/usr/bin/env bash
#SBATCH -J 'generate_accuracy_barplots'
#SBATCH -o outputs/slurm-%j-generate_accuracy_plots.out
#SBATCH -p all
#SBATCH -t 100
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

module load anaconda/4.4.0
source activate thesis
module load cudnn/cuda-9.0/7.0.3
module load cudatoolkit/9.0

python /home/cc27/Thesis/generalized_schema_learning/analysis/generate_accuracy_bargraphs_multitrials.py --exp_folder=fixedfiller_gensymbolicstates_100000_1_AllQs/fixed_filler/ --exp_title="Limited Filler Training" --exp_title2="Tested with Previously Seen Fillers" --trial_nums=[0,1,2] --split_queries="QDessert,QDrink,QEmcee,QFriend,QPoet,QSubject" --chance_rate=0.023  --epochs_dict='{"RNN-LN":1000, "LSTM-LN":1000, "RNN-LN-FW":50, "NTM2":50}'

python /home/cc27/Thesis/generalized_schema_learning/analysis/generate_accuracy_bargraphs_multitrials.py --exp_folder=fixedfiller_gensymbolicstates_100000_1_testunseen_AllQs/fixed_filler/ --exp_title="Limited Filler Training" --exp_title2="Tested with Previously Unseen Fillers" --trial_nums=[0,1,2] --split_queries="QDessert,QDrink,QEmcee,QFriend,QPoet,QSubject" --chance_rate=0.023  --epochs_dict='{"RNN-LN":3000, "LSTM-LN":200, "RNN-LN-FW":50, "NTM2":50}'

python /home/cc27/Thesis/generalized_schema_learning/analysis/generate_accuracy_bargraphs_multitrials.py --exp_folder=variablefiller_gensymbolicstates_100000_1_testunseen_AllQs/variable_filler/ --exp_title="Unlimited Filler Training" --exp_title2="Tested with Previously Unseen Fillers" --trial_nums=[0,1,2] --split_queries="QDessert,QDrink,QEmcee,QFriend,QPoet,QSubject" --chance_rate=0.013  --epochs_dict='{"RNN-LN":30000, "LSTM-LN":30000, "RNN-LN-FW":30000, "NTM2":30000}'

python /home/cc27/Thesis/generalized_schema_learning/analysis/generate_accuracy_bargraphs_multitrials.py --exp_folder=variablefiller_gensymbolicstates_100000_1_testunseen_QSubject/variable_filler/ --exp_title="Unlimited Filler Training" --exp_title2="Tested with Previously Unseen Fillers, QSubject Only" --trial_nums=[0,1,2] --chance_rate=0.013  --epochs_dict='{"RNN-LN":3000, "LSTM-LN":3000, "RNN-LN-FW":3000, "NTM2":3000}'
