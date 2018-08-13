#!/usr/bin/env bash
#SBATCH -J 'write_experiments'
#SBATCH -o outputs/slurm-%j-write_experiments.out
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

echo "Writing CSW experiments."
module load anaconda/4.4.0
source activate thesis
module load cudnn/cuda-9.0/7.0.3
module load cudatoolkit/9.0

echo "*************************************************************************"
python /home/cc27/Thesis/generalized_schema_learning/experiment_creators/experiment_writer.py --exp_name=fixedfiller_gensymbolicstates_100000_1 --gen_type=NONE
echo "*************************************************************************"
python /home/cc27/Thesis/generalized_schema_learning/experiment_creators/experiment_writer.py --exp_name=fixedfiller_gensymbolicstates_100000_1 --gen_type=TESTUNSEEN
echo "*************************************************************************"
python /home/cc27/Thesis/generalized_schema_learning/experiment_creators/experiment_writer.py --exp_name=variablefiller_gensymbolicstates_100000_1 --gen_type=TESTUNSEEN
echo "*************************************************************************"
python /home/cc27/Thesis/generalized_schema_learning/experiment_creators/experiment_writer.py --exp_name=fixedfiller_gensymbolicstates_100000_1 --gen_type=TESTUNSEEN --poss_qs=Subject
echo "*************************************************************************"
python /home/cc27/Thesis/generalized_schema_learning/experiment_creators/experiment_writer.py --exp_name=variablefiller_gensymbolicstates_100000_1 --gen_type=TESTUNSEEN --poss_qs=Subject,Friend
echo "*************************************************************************"
python /home/cc27/Thesis/generalized_schema_learning/experiment_creators/experiment_writer.py --exp_name=variablefiller_gensymbolicstates_100000_1 --gen_type=TESTUNSEEN --poss_qs=Subject
