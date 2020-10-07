#!/usr/bin/env bash
#SBATCH -J 'rnnln'
#SBATCH -o outputs/slurm-%j-experiment2rnnln.out
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

python /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=test --exp_name=fixedfiller_gensymbolicstates_100000_1_testunseen_AllQs --filler_type=fixed_filler --model_name=RNN-LN --trial_num=0 --test_filename="test.p"
echo "Done running predictions."
