#!/usr/bin/env bash
#SBATCH -J 'get_hardcoded_indices'
#SBATCH -o outputs/slurm-%j-get_hardcoded_indices.out
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

echo "Save split data."
module load anaconda/4.4.0
source activate thesis
module load cudnn/cuda-9.0/7.0.3
module load cudatoolkit/9.0

echo "*************************************************************************"
python /home/cc27/Thesis/generalized_schema_learning/experiment_creators/get_hardcoded_indices.py variablefiller_gensymbolicstates_100000_1_testunseen_AllQs "['EmceeFillerTest','QSubject','QDrink_bought','Order_dessert','BEGIN','Too_expensive','FriendFillerTrain','PersonFillerTest','Subject_performs','Say_goodbye','Emcee_intro','Sit_down','?','Order_drink','PoetFillerTrain','END','zzz','Poet_performs','QFriend','DrinkFillerTest','PoetFillerTest','QDessert_bought','QPoet','QEmcee','FriendFillerTest','EmceeFillerTrain','DessertFillerTrain','DrinkFillerTrain','PersonFillerTrain','DessertFillerTest','Subject_declines']"
echo "*************************************************************************"
python /home/cc27/Thesis/generalized_schema_learning/experiment_creators/get_hardcoded_indices.py variablefiller_gensymbolicstates_100000_1_testunseen_QSubjectQFriend "['EmceeFillerTest','QSubject','Order_dessert','BEGIN','Too_expensive','FriendFillerTrain','PersonFillerTest','Subject_performs','Say_goodbye','Emcee_intro','Sit_down','?','Order_drink','PoetFillerTrain','END','zzz','Poet_performs','QFriend','DrinkFillerTest','PoetFillerTest','FriendFillerTest','EmceeFillerTrain','DessertFillerTrain','DrinkFillerTrain','PersonFillerTrain','DessertFillerTest','Subject_declines']"
echo "*************************************************************************"
python /home/cc27/Thesis/generalized_schema_learning/experiment_creators/get_hardcoded_indices.py variablefiller_gensymbolicstates_100000_1_testunseen_QSubject "['EmceeFillerTest', 'QSubject', 'Order_dessert', 'BEGIN', 'Too_expensive', 'FriendFillerTrain', 'PersonFillerTest', 'Subject_performs', 'Say_goodbye', 'Emcee_intro', 'Sit_down', '?', 'Order_drink', 'PoetFillerTrain', 'END', 'zzz', 'Poet_performs', 'DrinkFillerTest', 'PoetFillerTest', 'FriendFillerTest', 'EmceeFillerTrain', 'DessertFillerTrain', 'DrinkFillerTrain', 'PersonFillerTrain', 'DessertFillerTest', 'Subject_declines']"
echo "*************************************************************************"
python /home/cc27/Thesis/generalized_schema_learning/experiment_creators/get_hardcoded_indices.py fixedfiller_gensymbolicstates_100000_1_AllQs "['Sarah', 'tea', 'sorbet', 'coffee', 'mousse', 'juice', 'QDrink_bought', 'pastry', 'Jane', 'John', 'milk', 'Order_dessert', 'BEGIN', 'espresso', 'Subject_performs', 'Olivia', 'Anna', 'Say_goodbye', 'Emcee_intro', 'Sit_down', '?', 'Order_drink', 'Pradeep', 'QSubject', 'END', 'zzz', 'Poet_performs', 'cheesecake', 'QFriend', 'Bill', 'QDessert_bought', 'chocolate', 'water', 'cookie', 'QEmcee', 'cupcake', 'latte', 'QPoet', 'Will', 'Julian', 'candy', 'cake', 'Mariko', 'Subject_declines']"
echo "*************************************************************************"
python /home/cc27/Thesis/generalized_schema_learning/experiment_creators/get_hardcoded_indices.py fixedfiller_gensymbolicstates_100000_1_testunseen_AllQs "['Sarah', 'tea', 'sorbet', 'coffee', 'mousse', 'juice', 'QDrink_bought', 'pastry', 'Jane', 'John', 'milk', 'Order_dessert', 'BEGIN', 'espresso', 'Subject_performs', 'Olivia', 'Anna', 'Say_goodbye', 'Emcee_intro', 'Sit_down', '?', 'Order_drink', 'Pradeep', 'QSubject', 'END', 'zzz', 'Poet_performs', 'cheesecake', 'QFriend', 'Bill', 'QDessert_bought', 'chocolate', 'water', 'cookie', 'QEmcee', 'cupcake', 'latte', 'QPoet', 'Will', 'Julian', 'candy', 'cake', 'Mariko', 'Subject_declines']"
echo "*************************************************************************"
python /home/cc27/Thesis/generalized_schema_learning/experiment_creators/get_hardcoded_indices.py fixedfiller_gensymbolicstates_100000_1_testunseen_QSubject "['Sarah', 'tea', 'sorbet', 'coffee', 'mousse', 'juice', 'pastry', 'Jane', 'John', 'milk', 'Order_dessert', 'BEGIN', 'espresso', 'Subject_performs', 'Olivia', 'Anna', 'Say_goodbye', 'Emcee_intro', 'Sit_down', '?', 'Order_drink', 'Pradeep', 'QSubject', 'END', 'zzz', 'Poet_performs', 'cheesecake', 'Bill', 'chocolate', 'water', 'cookie', 'cupcake', 'latte', 'Will', 'Julian', 'candy', 'cake', 'Mariko', 'Subject_declines']"
