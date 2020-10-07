#!/usr/bin/env bash
#SBATCH -J 'runtrain_variablefillergensymbolicstates1000001testunseenAllQs_NTM2_variablefiller_trial1'
#SBATCH -o outputs/slurm-%j-runtrain_variablefillergensymbolicstates1000001testunseenAllQs_NTM2_variablefiller_trial1.out
#SBATCH -p all
#SBATCH -t 8640
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

echo "Run CSW variablefiller_gensymbolicstates_100000_1_testunseen_AllQs experiment with NTM2"
module load anaconda/4.4.0
source activate thesis
module load cudnn/cuda-9.0/7.0.3
module load cudatoolkit/9.0
date
echo "***************************************** Running CSW variablefiller_gensymbolicstates_100000_1_testunseen_AllQs Training with NTM2 *****************************************"
python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=variablefiller_gensymbolicstates_100000_1_testunseen_AllQs --filler_type=variable_filler --model_name=NTM2 --num_epochs=7500 --trial_num=1
date
echo "Finished"
