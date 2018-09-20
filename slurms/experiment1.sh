#!/usr/bin/env bash
#SBATCH -J 'experiment1'
#SBATCH -o outputs/slurm-%j-experiment1.out
#SBATCH -p all
#SBATCH -t 1000
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:1

module load anaconda/4.4.0
source activate thesis
module load cudnn/cuda-9.0/7.0.3
module load cudatoolkit/9.0

python /home/cc27/Thesis/generalized_schema_learning/experiment_creators/experiment_writer.py --exp_name=fixedfiller_gensymbolicstates_100000_1 --gen_type=NONE
echo "Done writing inputs."

python /home/cc27/Thesis/generalized_schema_learning/experiment_creators/create_embedding.py fixedfiller_gensymbolicstates_100000_1_AllQs
echo "Done generating word vectors."

python /home/cc27/Thesis/generalized_schema_learning/experiment_creators/split_data.py fixedfiller_gensymbolicstates_100000_1_AllQs "['QDessert_bought','QDrink_bought','QEmcee','QFriend','QPoet','QSubject']" "['Sarah', 'tea', 'sorbet', 'coffee', 'mousse', 'juice', 'QDrink_bought', 'pastry', 'latte', 'Jane', 'John', 'milk', 'Order_dessert', 'BEGIN', 'Too_expensive', 'espresso', 'Subject_performs', 'Olivia', 'Anna', 'Say_goodbye', 'Emcee_intro', 'Sit_down', '?', 'Order_drink', 'Pradeep', 'QSubject', 'END', 'zzz', 'Poet_performs', 'cheesecake', 'QFriend', 'Bill', 'QDessert_bought', 'chocolate', 'water', 'cookie', 'QEmcee', 'cupcake', 'QPoet', 'Will', 'Julian', 'candy', 'cake', 'Mariko', 'Subject_declines']"
echo "Done generating split datasets."

python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=fixedfiller_gensymbolicstates_100000_1_AllQs --filler_type=fixed_filler --model_name=RNN-LN --num_epochs=2000 --trial_num=0
python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=fixedfiller_gensymbolicstates_100000_1_AllQs --filler_type=fixed_filler --model_name=RNN-LN-FW --num_epochs=2000 --trial_num=0
python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=fixedfiller_gensymbolicstates_100000_1_AllQs --filler_type=fixed_filler --model_name=NTM2 --num_epochs=2000 --trial_num=0
python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=fixedfiller_gensymbolicstates_100000_1_AllQs --filler_type=fixed_filler --model_name=LSTM-LN --num_epochs=2000 --trial_num=0
echo "Done training networks."

python /home/cc27/Thesis/generalized_schema_learning/analysis/generate_accuracy_plots.py --exp_folder=fixedfiller_gensymbolicstates_100000_1_AllQs/fixed_filler/ --exp_title="Fixed Random Representation" --exp_title2="AllQs, Previously Seen Fillers" --num_smooth=10 --trial_num=0 --split_queries="QDessert,QDrink,QEmcee,QFriend,QPoet,QSubject" --chance_rate=0.023 # For accuracy vs epochs.
python /home/cc27/Thesis/generalized_schema_learning/analysis/generate_accuracy_bargraphs_multitrials.py --exp_folder=fixedfiller_gensymbolicstates_100000_1_AllQs/fixed_filler/ --exp_title="Fixed Random Representation" --exp_title2="AllQs, Previously Seen Fillers" --save="True" --chance_rate=0.023 --epochs_dict='{"RNN-LN":1000, "LSTM-LN":1000, "RNN-LN-FW":50, "NTM2":50}' --trial_nums="[0,1,2]" # For bar charts.
echo "Done generating plots."
