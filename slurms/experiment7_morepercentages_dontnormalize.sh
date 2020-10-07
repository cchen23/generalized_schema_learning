#!/usr/bin/env bash
#SBATCH -J 'statret'
#SBATCH -o outputs/slurm-%j-experiment7_morepercentages.out
#SBATCH -p all
#SBATCH -t 100
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1

module load anaconda/4.4.0
source activate thesis
module load cudnn/cuda-9.0/7.0.3
module load cudatoolkit/9.0

## Generate stories.
python -u /home/cc27/Thesis/generalized_schema_learning/experiment_creators/story_probestatisticsretention_generator.py 100 False
python -u /home/cc27/Thesis/generalized_schema_learning/experiment_creators/story_probestatisticsretention_generator.py 75 False
python -u /home/cc27/Thesis/generalized_schema_learning/experiment_creators/story_probestatisticsretention_generator.py 50 False
python -u /home/cc27/Thesis/generalized_schema_learning/experiment_creators/story_probestatisticsretention_generator.py 25 False
python -u /home/cc27/Thesis/generalized_schema_learning/experiment_creators/story_probestatisticsretention_generator.py 0 False

#python /home/cc27/Thesis/generalized_schema_learning/analysis/generate_accuracy_bargraphs_multitrials.py --exp_folder=probestatisticsretention_percentageindistribution100/fixed_filler --exp_title="Special Filler Distribution" --exp_title2="100 percent" --save="True" --split_queries="QEmcee,QFriend,QPoet,QSubject" --chance_rate=0.023 --epochs_dict='{"RNN-LN":2500, "LSTM-LN":1250, "RNN-LN-FW":1000, "NTM2":25}' --trial_nums="[0]" # For bar charts.
#
#python /home/cc27/Thesis/generalized_schema_learning/analysis/generate_accuracy_bargraphs_multitrials.py --exp_folder=probestatisticsretention_percentageindistribution75/fixed_filler --exp_title="Special Filler Distribution" --exp_title2="75 percent" --save="True" --split_queries="QEmcee,QFriend,QPoet,QSubject" --chance_rate=0.023 --epochs_dict='{"RNN-LN":2500, "LSTM-LN":2500, "RNN-LN-FW":2000, "NTM2":50}' --trial_nums="[0]" # For bar charts.
#
#python /home/cc27/Thesis/generalized_schema_learning/analysis/generate_accuracy_bargraphs_multitrials.py --exp_folder=probestatisticsretention_percentageindistribution50/fixed_filler --exp_title="Special Filler Distribution" --exp_title2="50 percent" --save="True" --split_queries="QEmcee,QFriend,QPoet,QSubject" --chance_rate=0.023 --epochs_dict='{"RNN-LN":6250, "LSTM-LN":2500, "RNN-LN-FW":2500, "NTM2":75}' --trial_nums="[0]" # For bar charts.
#
#python /home/cc27/Thesis/generalized_schema_learning/analysis/generate_accuracy_bargraphs_multitrials.py --exp_folder=probestatisticsretention_percentageindistribution25/fixed_filler --exp_title="Special Filler Distribution" --exp_title2="25 percent" --save="True" --split_queries="QEmcee,QFriend,QPoet,QSubject" --chance_rate=0.023 --epochs_dict='{"RNN-LN":6875, "LSTM-LN":1250, "RNN-LN-FW":1000, "NTM2":25}' --trial_nums="[0]" # For bar charts.
