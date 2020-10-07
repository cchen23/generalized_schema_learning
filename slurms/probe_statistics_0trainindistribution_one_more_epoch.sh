#!/usr/bin/env bash
#SBATCH -J "oneepoch"
#SBATCH -o outputs/slurm-%j-probe_statistics.out
#SBATCH -p all
#SBATCH -t 1300
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=cc27@alumni.princeton.edu
module load anaconda/4.4.0
source activate thesis
module load cudnn/cuda-9.0/7.0.3
module load cudatoolkit/9.0
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=0 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=1 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=2 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=3 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=4 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=5 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=6 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=7 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=8 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=9 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=10 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=11 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=12 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=13 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=14 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=15 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=16 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=17 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=18 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=19 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=20 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=21 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=22 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=23 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=24 --num_epochs=1
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=train --exp_name=probestatisticsretention_percentageindistribution0_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=25 --num_epochs=1
