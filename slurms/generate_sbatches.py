import os
import subprocess
# Info: https://slurm.schedmd.com/sbatch.html
script = os.path.join("/", "home", "cc27", "Thesis", "generalized_schema_learning", "run_experiment.py")
output = os.path.join("/" "home", "cc27", "Thesis", "generalized_schema_learning", "slurms", "outputs", "{experiment_name}_%j.out")
job_name = "a_{num_persons_per_category}"

sb_cmd = """#!/usr/bin/bash

module load anaconda/4.4.0
source activate thesis
module load cudnn/cuda-9.0/7.0.3
module load cudatoolkit/9.0

python {script} --function=train --exp_name={experiment_name} --filler_type=fixed_filler --model_name={model_name} --num_epochs=1000 --trial_num={trial_num}
"""
num_persons_per_category_options = [10, 100, 1000]
model_name_options = ["NTM2", "RNN-LN-FW", "LSTM-LN", "RNN-LN"]
for trial_num in range(3):
    for num_persons_per_category in num_persons_per_category_options:
        for model_name in model_name_options:
            experiment_name = "generate_train3roles_testnewrole_withunseentestfillers_%dpersonspercategory_24000train_120test" % num_persons_per_category
            sbproc = subprocess.Popen(["sbatch",
                                       "--output=" + output.format(experiment_name=experiment_name),
                                       "--job-name=" + job_name.format(num_persons_per_category=num_persons_per_category),
                                       "--partition=all",
                                       "--time=8640",
                                       "--gres=gpu:1",
                                       "--nodes=1",
                                       "--ntasks-per-node=4",
                                       "--ntasks-per-socket=2"
                                       ],
                                      stdin=subprocess.PIPE)
            thiscmd = sb_cmd.format(script=script, experiment_name=experiment_name, model_name=model_name, trial_num=trial_num)
            print(thiscmd)
            sbproc.communicate(thiscmd)
