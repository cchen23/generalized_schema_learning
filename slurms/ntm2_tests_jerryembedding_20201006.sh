#!/usr/bin/env bash
#SBATCH -J 'experiment3'
#SBATCH -o outputs/slurm-%j-ntm2testjerryembedding.out
#SBATCH -p all
#SBATCH -t 1000
#SBATCH -N 1
#SBATCH --gres=gpu:1

module load anaconda/4.4.0
source activate thesis
module load cudnn/cuda-9.0/7.0.3
module load cudatoolkit/9.0

python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=variablefiller_gensymbolicstates_100000_1_testunseen_AllQs --filler_type=variable_filler --model_name=NTM2 --num_epochs=30000 --trial_num=5 --embedding_type=old

#python /home/cc27/Thesis/generalized_schema_learning/analysis/generate_accuracy_plots.py --exp_folder=variablefiller_gensymbolicstates_100000_1_testunseen_AllQs/variable_filler/ --exp_title="Variable Random Representation" --exp_title2="AllQs, Previously Unseen Fillers" --trial_num=0 --split_queries="QDessert,QDrink,QEmcee,QFriend,QPoet,QSubject" --chance_rate=0.013 # For accuracy vs epochs.
#python /home/cc27/Thesis/generalized_schema_learning/analysis/generate_accuracy_bargraphs_multitrials.py --exp_folder=variablefiller_gensymbolicstates_100000_1_testunseen_AllQs/variable_filler/ --exp_title="Variable Random Representation" --exp_title2="AllQs, Previously Unseen Fillers" --save="True" --split_queries="QDessert,QDrink,QEmcee,QFriend,QPoet,QSubject" --chance_rate=0.023 --epochs_dict='{"RNN-LN":30000, "LSTM-LN":30000, "RNN-LN-FW":30000, "NTM2":30000}' --trial_nums="[0,1,2]" # For bar charts.
#echo "Done generating plots."
