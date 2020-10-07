#!/usr/bin/env bash
#SBATCH -J 'runanalysis_variablefillergensymbolicstates1000001testunseenAllQs_variablefiller_test_analyze_trial0'
#SBATCH -o outputs/slurm-%j-runanalysis_variablefillergensymbolicstates1000001testunseenAllQs_variablefiller_test_analyze_trial0.out
#SBATCH -p all
#SBATCH -t 500
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

echo "Run CSW variablefiller_gensymbolicstates_100000_1_testunseen_AllQs analysis with all architectures"
module load anaconda/4.4.0
source activate thesis
module load cudnn/cuda-9.0/7.0.3
module load cudatoolkit/9.0
echo "***************************************** Running CSW variablefiller_gensymbolicstates_100000_1_testunseen_AllQs Testing with NTM2 *****************************************"
python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=analyze --exp_name=variablefiller_gensymbolicstates_100000_1_testunseen_AllQs --filler_type=variable_filler --model_name=NTM2 --test_filename=test_analyze.p --trial_num=0
echo "Finished"
