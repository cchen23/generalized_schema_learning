#!/usr/bin/env bash
#SBATCH -J 'experiment3'
#SBATCH -o outputs/slurm-%j-experiment3.out
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

#python /home/cc27/Thesis/generalized_schema_learning/experiment_creators/experiment_writer.py --exp_name=variablefiller_gensymbolicstates_100000_1 --gen_type=TESTUNSEEN
#python /home/cc27/Thesis/generalized_schema_learning/experiment_creators/experiment_writer.py --exp_name=variablefiller_gensymbolicstates_100000_1 --gen_type=TESTUNSEEN --poss_qs=Subject
#echo "Done writing inputs."

#python /home/cc27/Thesis/generalized_schema_learning/experiment_creators/create_embedding.py variablefiller_gensymbolicstates_100000_1_testunseen_AllQs
#python /home/cc27/Thesis/generalized_schema_learning/experiment_creators/create_embedding.py variablefiller_gensymbolicstates_100000_1_testunseen_QSubject
#echo "Done generating word vectors."

#python /home/cc27/Thesis/generalized_schema_learning/experiment_creators/split_data.py variablefiller_gensymbolicstates_100000_1_testunseen_AllQs "['QDessert_bought','QDrink_bought','QEmcee','QFriend','QPoet','QSubject']" "['EmceeFillerTest','QSubject','QDrink_bought','Order_dessert','BEGIN','Too_expensive','FriendFillerTrain','PersonFillerTest','Subject_performs','Say_goodbye','Emcee_intro','Sit_down','?','Order_drink','PoetFillerTrain','END','zzz','Poet_performs','QFriend','DrinkFillerTest','PoetFillerTest','QDessert_bought','QPoet','QEmcee','FriendFillerTest','EmceeFillerTrain','DessertFillerTrain','DrinkFillerTrain','PersonFillerTrain','DessertFillerTest','Subject_declines']"
#echo "Done generating split datasets."

# python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=variablefiller_gensymbolicstates_100000_1_testunseen_AllQs --filler_type=variable_filler --model_name=RNN-LN --num_epochs=30000 --trial_num=0
# python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=variablefiller_gensymbolicstates_100000_1_testunseen_AllQs --filler_type=variable_filler --model_name=RNN-LN-FW --num_epochs=30000 --trial_num=0
# python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=variablefiller_gensymbolicstates_100000_1_testunseen_AllQs --filler_type=variable_filler --model_name=LSTM-LN --num_epochs=30000 --trial_num=0
# python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=variablefiller_gensymbolicstates_100000_1_testunseen_AllQs --filler_type=variable_filler --model_name=NTM2 --num_epochs=30000 --trial_num=0
# python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=variablefiller_gensymbolicstates_100000_1_testunseen_QSubject --filler_type=variable_filler --model_name=RNN-LN --num_epochs=3000 --trial_num=0
# python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=variablefiller_gensymbolicstates_100000_1_testunseen_QSubject --filler_type=variable_filler --model_name=RNN-LN-FW --num_epochs=3000 --trial_num=0
# python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=variablefiller_gensymbolicstates_100000_1_testunseen_QSubject --filler_type=variable_filler --model_name=LSTM-LN --num_epochs=3000 --trial_num=0
# python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=variablefiller_gensymbolicstates_100000_1_testunseen_QSubject --filler_type=variable_filler --model_name=NTM2 --num_epochs=3000 --trial_num=0
# echo "Done training networks."

python /home/cc27/Thesis/generalized_schema_learning/analysis/generate_accuracy_plots.py --exp_folder=variablefiller_gensymbolicstates_100000_1_testunseen_AllQs/variable_filler/ --exp_title="Variable Random Representation" --exp_title2="AllQs, Previously Unseen Fillers" --trial_num=0 --split_queries="QDessert,QDrink,QEmcee,QFriend,QPoet,QSubject"
python /home/cc27/Thesis/generalized_schema_learning/analysis/generate_accuracy_plots.py --exp_folder=variablefiller_gensymbolicstates_100000_1_testunseen_QSubject/variable_filler/ --exp_title="Variable Random Representation" --exp_title2="QSubject, Previously Unseen Fillers" --trial_num=0
echo "Done generating plots."
