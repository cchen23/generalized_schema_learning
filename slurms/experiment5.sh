#!/usr/bin/env bash
#SBATCH -J 'f_0'
#SBATCH -o outputs/slurm-%j-experiment5.out
#SBATCH -p all
#SBATCH -t 8640
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:1

module load anaconda/4.4.0
source activate thesis
module load cudnn/cuda-9.0/7.0.3
module load cudatoolkit/9.0

# python /home/cc27/Thesis/generalized_schema_learning/experiment_creators/storyv2_generator.py

#python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=storyv2_train20000_AllQs --filler_type=variable_filler --model_name=RNN-LN-FW --num_epochs=5000 --trial_num=0
#
#python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=test_split --exp_name=generate_train3roles_testnewrole_withunseentestfillers_1000personspercategory_24000train_120test --filler_type=fixed_filler --model_name=NTM2 --trial_num=0
#python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=test_split --exp_name=generate_train3roles_testnewrole_withunseentestfillers_1000personspercategory_24000train_120test --filler_type=fixed_filler --model_name=RNN-LN-FW --trial_num=0
#python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=test_split --exp_name=generate_train3roles_testnewrole_withunseentestfillers_1000personspercategory_24000train_120test --filler_type=fixed_filler --model_name=RNN-LN --trial_num=0
#python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=test_split --exp_name=generate_train3roles_testnewrole_withunseentestfillers_1000personspercategory_24000train_120test --filler_type=fixed_filler --model_name=LSTM-LN --trial_num=0
#
#python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=test_split --exp_name=generate_train3roles_testnewrole_withunseentestfillers_100personspercategory_24000train_120test --filler_type=fixed_filler --model_name=NTM2 --trial_num=0
#python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=test_split --exp_name=generate_train3roles_testnewrole_withunseentestfillers_100personspercategory_24000train_120test --filler_type=fixed_filler --model_name=RNN-LN-FW --trial_num=0
#python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=test_split --exp_name=generate_train3roles_testnewrole_withunseentestfillers_100personspercategory_24000train_120test --filler_type=fixed_filler --model_name=RNN-LN --trial_num=0
#python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=test_split --exp_name=generate_train3roles_testnewrole_withunseentestfillers_100personspercategory_24000train_120test --filler_type=fixed_filler --model_name=LSTM-LN --trial_num=0
#
#python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=test_split --exp_name=generate_train3roles_testnewrole_withunseentestfillers_10personspercategory_24000train_120test --filler_type=fixed_filler --model_name=NTM2 --trial_num=0
#python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=test_split --exp_name=generate_train3roles_testnewrole_withunseentestfillers_10personspercategory_24000train_120test --filler_type=fixed_filler --model_name=RNN-LN-FW --trial_num=0
#python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=test_split --exp_name=generate_train3roles_testnewrole_withunseentestfillers_10personspercategory_24000train_120test --filler_type=fixed_filler --model_name=RNN-LN --trial_num=0
#python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=test_split --exp_name=generate_train3roles_testnewrole_withunseentestfillers_10personspercategory_24000train_120test --filler_type=fixed_filler --model_name=LSTM-LN --trial_num=0

python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=generate_train3roles_testnewrole_withunseentestfillers_1000personspercategory_24000train_120test_add05fillers --filler_type=fixed_filler --model_name=NTM2 --trial_num=0
generate_train3roles_testnewrole_withunseentestfillers_1000personspercategory_24000train_120test_add05fillers
