import os

exp_name_template = "probestatisticsretention_percentageindistribution{percentage_train_indistribution}_normalizefillerdistributionFalse"
filler_type = "fixed_filler"
model_name = "NTM2"

percentage_train_indistribution_options = [0, 50]

for percentage_train_indistribution in percentage_train_indistribution_options:
    f = open('probe_statistics_{percentage_train_indistribution}trainindistribution_one_more_epoch.sh'.format(percentage_train_indistribution=percentage_train_indistribution), 'wb')
    f.write('#!/usr/bin/env bash\n')
    f.write('#SBATCH -J "oneepoch"\n')
    f.write('#SBATCH -o outputs/slurm-%j-probe_statistics.out\n')
    f.write('#SBATCH -p all\n')
    f.write('#SBATCH -t 1300\n')
    f.write('#SBATCH -N 1\n')
    f.write('#SBATCH --ntasks-per-node=4\n')
    f.write('#SBATCH --gres=gpu:1\n')
    f.write('#SBATCH --mail-type=FAIL,BEGIN,END\n')
    f.write('#SBATCH --mail-user=cc27@alumni.princeton.edu\n')

    f.write('module load anaconda/4.4.0\n')
    f.write('source activate thesis\n')
    f.write('module load cudnn/cuda-9.0/7.0.3\n')
    f.write('module load cudatoolkit/9.0\n')

    exp_name = exp_name_template.format(percentage_train_indistribution=percentage_train_indistribution)
    checkpoint_dirs = os.listdir(os.path.join('/home/cc27/Thesis/generalized_schema_learning/checkpoints/', exp_name, 'fixed_filler', 'NTM2'))
    trial_nums = [int(checkpoint_dir.split('trial')[-1]) for checkpoint_dir in checkpoint_dirs]
    for trial_num in trial_nums:
        f.write('python -u {script} --function={function} --exp_name={exp_name} --filler_type={filler_type} --model_name={model_name} --trial_num={trial_num} --num_epochs=1\n'.format(
            script='/home/cc27/Thesis/generalized_schema_learning/run_experiment.py',
            function='train',
            exp_name=exp_name,
            filler_type='fixed_filler',
            model_name='NTM2',
            trial_num=trial_num,
        ))
    f.close()
