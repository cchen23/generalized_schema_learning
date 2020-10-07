#!/usr/bin/env bash
#SBATCH -J "probe"
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
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=probe_statistics --exp_name=probestatisticsretention_percentageindistribution50_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=10 --test_filename=test_QEmcee_replaceemcee.p --set_to_zero=True
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=probe_statistics --exp_name=probestatisticsretention_percentageindistribution50_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=10 --test_filename=test_QEmcee_replaceemcee.p --set_to_zero=False
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=probe_statistics --exp_name=probestatisticsretention_percentageindistribution50_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=11 --test_filename=test_QEmcee_replaceemcee.p --set_to_zero=True
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=probe_statistics --exp_name=probestatisticsretention_percentageindistribution50_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=11 --test_filename=test_QEmcee_replaceemcee.p --set_to_zero=False
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=probe_statistics --exp_name=probestatisticsretention_percentageindistribution50_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=12 --test_filename=test_QEmcee_replaceemcee.p --set_to_zero=True
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=probe_statistics --exp_name=probestatisticsretention_percentageindistribution50_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=12 --test_filename=test_QEmcee_replaceemcee.p --set_to_zero=False
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=probe_statistics --exp_name=probestatisticsretention_percentageindistribution50_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=13 --test_filename=test_QEmcee_replaceemcee.p --set_to_zero=True
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=probe_statistics --exp_name=probestatisticsretention_percentageindistribution50_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=13 --test_filename=test_QEmcee_replaceemcee.p --set_to_zero=False
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=probe_statistics --exp_name=probestatisticsretention_percentageindistribution50_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=14 --test_filename=test_QEmcee_replaceemcee.p --set_to_zero=True
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=probe_statistics --exp_name=probestatisticsretention_percentageindistribution50_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=14 --test_filename=test_QEmcee_replaceemcee.p --set_to_zero=False
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=probe_statistics --exp_name=probestatisticsretention_percentageindistribution50_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=15 --test_filename=test_QEmcee_replaceemcee.p --set_to_zero=True
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=probe_statistics --exp_name=probestatisticsretention_percentageindistribution50_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=15 --test_filename=test_QEmcee_replaceemcee.p --set_to_zero=False
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=probe_statistics --exp_name=probestatisticsretention_percentageindistribution50_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=16 --test_filename=test_QEmcee_replaceemcee.p --set_to_zero=True
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=probe_statistics --exp_name=probestatisticsretention_percentageindistribution50_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=16 --test_filename=test_QEmcee_replaceemcee.p --set_to_zero=False
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=probe_statistics --exp_name=probestatisticsretention_percentageindistribution50_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=17 --test_filename=test_QEmcee_replaceemcee.p --set_to_zero=True
python -u /home/cc27/Thesis/generalized_schema_learning/run_experiment.py --function=probe_statistics --exp_name=probestatisticsretention_percentageindistribution50_normalizefillerdistributionFalse --filler_type=fixed_filler --model_name=NTM2 --trial_num=17 --test_filename=test_QEmcee_replaceemcee.p --set_to_zero=False
