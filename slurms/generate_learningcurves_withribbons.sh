#!/usr/bin/env bash
#SBATCH -J 'learningcurves'
#SBATCH -o outputs/slurm-%j-trainingcurves-ribbons.out
#SBATCH -p all
#SBATCH -t 100
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:1

module load anaconda/4.4.0
source activate thesis
module load cudnn/cuda-9.0/7.0.3
module load cudatoolkit/9.0

python /home/cc27/Thesis/generalized_schema_learning/analysis/generate_accuracy_plots_multitrials.py --exp_folder=fixedfiller_gensymbolicstates_100000_1_AllQs/fixed_filler/ --xlim=1000 --exp_title="Fixed Random Representation" --exp_title2="AllQs, Previously Seen Fillers" --num_smooth=10 --trial_nums=[0,1,2] --split_queries="QDessert,QDrink,QEmcee,QFriend,QPoet,QSubject" --chance_rate=0.023 --save=True # For accuracy vs epochs.
python /home/cc27/Thesis/generalized_schema_learning/analysis/generate_accuracy_plots_multitrials.py --exp_folder=fixedfiller_gensymbolicstates_100000_1_testunseen_AllQs/fixed_filler/ --xlim=1000 --exp_title="Fixed Random Representation" --exp_title2="AllQs, Previously Unseen Fillers" --num_smooth=10 --trial_nums=[0,1,2] --split_queries="QDessert,QDrink,QEmcee,QFriend,QPoet,QSubject" --chance_rate=0.023 --save=True # For accuracy vs epochs.

python /home/cc27/Thesis/generalized_schema_learning/analysis/generate_accuracy_plots_multitrials.py --exp_folder=variablefiller_gensymbolicstates_100000_1_testunseen_AllQs/variable_filler/ --xlim=30000 --exp_title="Variable Random Representation" --exp_title2="AllQs, Previously Unseen Fillers" --trial_nums=[0,1,2] --split_queries="QDessert,QDrink,QEmcee,QFriend,QPoet,QSubject" --chance_rate=0.013 --save=True # For accuracy vs epochs.
python /home/cc27/Thesis/generalized_schema_learning/analysis/generate_accuracy_plots_multitrials.py --exp_folder=variablefiller_gensymbolicstates_100000_1_testunseen_QSubject/variable_filler/ --xlim=3000 --exp_title="Variable Random Representation" --exp_title2="QSubject, Previously Unseen Fillers" --trial_nums=[0,1,2] --chance_rate=0.013
