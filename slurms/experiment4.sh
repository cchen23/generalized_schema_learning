#!/usr/bin/env bash
#SBATCH -J 'experiment4'
#SBATCH -o outputs/slurm-%j-experiment4.out
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

# python /home/cc27/Thesis/generalized_schema_learning/experiment_creators/create_analysis_data.py variablefiller_gensymbolicstates_100000_1_testunseen_AllQs
# python /home/cc27/Thesis/generalized_schema_learning/experiment_creators/create_analysis_data.py variablefiller_gensymbolicstates_100000_1_testunseen_Subject
# echo "Generated analysis data."

# python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=analyze --exp_name=variablefiller_gensymbolicstates_100000_1_testunseen_Subject --filler_type=variable_filler --model_name=RNN-LN --test_filename=test_analyze.p --trial_num=0
# python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=analyze --exp_name=variablefiller_gensymbolicstates_100000_1_testunseen_Subject --filler_type=variable_filler --model_name=RNN-LN-FW --test_filename=test_analyze.p --trial_num=0
# python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=analyze --exp_name=variablefiller_gensymbolicstates_100000_1_testunseen_Subject --filler_type=variable_filler --model_name=LSTM-LN --test_filename=test_analyze.p --trial_num=0
# python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=analyze --exp_name=variablefiller_gensymbolicstates_100000_1_testunseen_Subject --filler_type=variable_filler --model_name=NTM2 --test_filename=test_analyze.p --trial_num=0
# python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=analyze --exp_name=variablefiller_gensymbolicstates_100000_1_testunseen_AllQs --filler_type=variable_filler --model_name=RNN-LN --test_filename=test_analyze.p --trial_num=0
# python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=analyze --exp_name=variablefiller_gensymbolicstates_100000_1_testunseen_AllQs --filler_type=variable_filler --model_name=RNN-LN-FW --test_filename=test_analyze.p --trial_num=0
# python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=analyze --exp_name=variablefiller_gensymbolicstates_100000_1_testunseen_AllQs --filler_type=variable_filler --model_name=LSTM-LN --test_filename=test_analyze.p --trial_num=0
# python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=analyze --exp_name=variablefiller_gensymbolicstates_100000_1_testunseen_AllQs --filler_type=variable_filler --model_name=NTM2 --test_filename=test_analyze.p --trial_num=0
# echo "Saved network states."
#
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py AllQs 30000 Dessert RNN-LN
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py AllQs 30000 Drink RNN-LN
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py AllQs 30000 Emcee RNN-LN
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py AllQs 30000 Friend RNN-LN
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py AllQs 30000 Poet RNN-LN
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py AllQs 30000 Subject RNN-LN

python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py AllQs 30000 Dessert RNN-LN-FW
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py AllQs 30000 Drink RNN-LN-FW
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py AllQs 30000 Emcee RNN-LN-FW
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py AllQs 30000 Friend RNN-LN-FW
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py AllQs 30000 Poet RNN-LN-FW
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py AllQs 30000 Subject RNN-LN-FW

python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py AllQs 30000 Dessert LSTM-LN
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py AllQs 30000 Drink LSTM-LN
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py AllQs 30000 Emcee LSTM-LN
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py AllQs 30000 Friend LSTM-LN
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py AllQs 30000 Poet LSTM-LN
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py AllQs 30000 Subject LSTM-LN

python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py AllQs 30000 Dessert NTM2
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py AllQs 30000 Drink NTM2
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py AllQs 30000 Emcee NTM2
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py AllQs 30000 Friend NTM2
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py AllQs 30000 Poet NTM2
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py AllQs 30000 Subject NTM2

python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py QSubject 3000 Dessert RNN-LN
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py QSubject 3000 Drink RNN-LN
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py QSubject 3000 Emcee RNN-LN
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py QSubject 3000 Friend RNN-LN
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py QSubject 3000 Poet RNN-LN
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py QSubject 3000 Subject RNN-LN

python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py QSubject 3000 Dessert RNN-LN-FW
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py QSubject 3000 Drink RNN-LN-FW
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py QSubject 3000 Emcee RNN-LN-FW
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py QSubject 3000 Friend RNN-LN-FW
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py QSubject 3000 Poet RNN-LN-FW
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py QSubject 3000 Subject RNN-LN-FW

python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py QSubject 3000 Dessert LSTM-LN
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py QSubject 3000 Drink LSTM-LN
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py QSubject 3000 Emcee LSTM-LN
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py QSubject 3000 Friend LSTM-LN
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py QSubject 3000 Poet LSTM-LN
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py QSubject 3000 Subject LSTM-LN

python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py QSubject 3000 Dessert NTM2
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py QSubject 3000 Drink NTM2
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py QSubject 3000 Emcee NTM2
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py QSubject 3000 Friend NTM2
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py QSubject 3000 Poet NTM2
python /home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py QSubject 3000 Subject NTM2
