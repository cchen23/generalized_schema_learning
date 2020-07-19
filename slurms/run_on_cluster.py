import numpy as np
import os
import subprocess
import time

# Info: https://slurm.schedmd.com/sbatch.html
function = "train"
exp_name_template = "probestatisticsretention_percentageindistribution{percentage_train_indistribution}_normalizefillerdistributionFalse"
filler_type = "fixed_filler"
num_epochs_dict = {"RNN-LN":2500, "LSTM-LN":1250, "NTM2":50, "RNN-LN-FW":1000}
model_names = ["NTM2"]
#percentage_train_indistribution_options = [100, 75, 50, 25, 0]
#percentage_train_indistribution_options = [0, 50, 100]
percentage_train_indistribution_options = [0, 50]
test_filenames = ["test_QFriend_replacefriend.p", "test_QPoet_replacepoet.p", "test_QSubject_replacesubject.p", "test_QEmcee_replaceemcee.p"]

#trial_nums = [0, 1, 2, 3]
#trial_nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#trial_nums = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
#trial_nums = [19]
trial_nums = range(25)
#percentage_train_indistribution_options = [90, 80, 60, 40, 20]
#percentage_train_indistribution_options = [100, 75, 50, 25, 0, 90, 80, 60, 40, 20]
script = "/home/cc27/Thesis/generalized_schema_learning/run_experiment.py"

sb_cmd = """#!/usr/bin/env bash
module load anaconda/4.4.0
source activate thesis
module load cudnn/cuda-9.0/7.0.3
module load cudatoolkit/9.0
"""
if function == "train":
    for trial_num in trial_nums:
        for percentage_train_indistribution in percentage_train_indistribution_options:
            exp_name = exp_name_template.format(percentage_train_indistribution=percentage_train_indistribution)
            output = "/home/cc27/Thesis/generalized_schema_learning/slurms/outputs/train_experiment{exp_name}_{model_name}_{function}_trial{trial_num}_start{datetime}_%j.txt"
            cmd = sb_cmd + "\npython -u {script} --function={function} --exp_name={exp_name} --filler_type={filler_type} --model_name={model_name} --trial_num={trial_num} --num_epochs={num_epochs}"
            for model_name in model_names:
                this_output = output.format(model_name=model_name, function=function, exp_name=exp_name, datetime=time.strftime("%y%m%d") + "_" + time.strftime("%H%M%D").replace("/",""), trial_num=trial_num)
                sbproc = subprocess.Popen(["sbatch",
                                           "--output=" + this_output,
                                           "-J", model_name,
                                           "-p", "all",
                                           "-t", "5000",
                                           "--gres", "gpu:1",
                                           "--mail-type", "FAIL,BEGIN,END",
                                           "--mail-user", "cc27@alumni.princeton.edu"
                                   ],
                                      stdin=subprocess.PIPE)
                thiscmd = cmd.format(script=script, function=function, exp_name=exp_name, filler_type=filler_type, model_name=model_name, trial_num=trial_num, num_epochs=num_epochs_dict[model_name])
                print(thiscmd)
                print('********')
                #print(this_output)
                sbproc.communicate(thiscmd)
elif function == "probe_statistics":
    i = 0
    for test_filename in test_filenames:
        for trial_num in trial_nums:
            for percentage_train_indistribution in percentage_train_indistribution_options:
                for model_name in model_names:
                    exp_name = exp_name_template.format(percentage_train_indistribution=percentage_train_indistribution)
                    sb_cmd += "\npython -u {script} --function={function} --exp_name={exp_name} --filler_type={filler_type} --model_name={model_name} --trial_num={trial_num} --test_filename={test_filename}"
                    output = "/home/cc27/Thesis/generalized_/schema_learning/slurms/outputs/probestatistics_experiment{exp_name}_{model_name}_{function}_trial{trial_num}_{datetime}_%j.txt"
                    sbproc = subprocess.Popen(["sbatch",
                                               "--output=" + output.format(model_name=model_name, function=function, exp_name=exp_name, test_filename=test_filename, datetime=time.strftime("%y%m%d") + "_" + time.strftime("%H%M%D").replace("/",""), trial_num=trial_num),
                                               "-J", model_name,
                                               "-p", "all",
                                               "-t", "1300",
                                               "-N", "1",
                                               "--ntasks-per-node", "4",
                                               "--gres", "gpu:1",
                                               "--mail-type", "FAIL, BEGIN, END",
                                               "--mail-user", "cc27@alumni.princeton.edu"
                                       ],
                                          stdin=subprocess.PIPE)
                    thiscmd = sb_cmd.format(script=script, function=function, exp_name=exp_name, filler_type=filler_type, model_name=model_name, trial_num=trial_num, test_filename=test_filename)
                    if i % 25 == 0:
                        #print(thiscmd)
                        print("******")
                        print(i)
                    i += 1
                    sbproc.communicate(thiscmd)
