import argparse
import subprocess
import time

# Info: https://slurm.schedmd.com/sbatch.html
parser = argparse.ArgumentParser()
parser.add_argument('--function', default='train')
parser.add_argument('--experiment-name')
parser.add_argument('--filler-type', default='fixed_filler')
parser.add_argument('--model-names', nargs='+')
parser.add_argument('--model-names-and-num-epochs',
                    nargs='+',
                    default=['NTM2_50', 'RNN-LN-FW_1000'],
                    #default=['RNN-LN_2500', 'LSTM-LN_1250', 'NTM2_50', 'RNN-LN-FW_1000'],
                    help='model names followed by number of training epochs (e.g. RNN-LN_2500 LSTM-LN_1250')
parser.add_argument('--test-filenames',
                    nargs='+',
                    default=['test.p', 'test_ambiguous_all.p', 'test_ambiguous_queried_role.p', 'test_flipped_distribution.p', 'test_unseen.p', 'test_unseen_flipped_distribution.p', 'test_unseen_no_addition_distribution.p'])
parser.add_argument('--trial-nums',
                    nargs='+',
                    default=[trial_num for trial_num in range(25)])
args = parser.parse_args()

script = "/home/cc27/Thesis/generalized_schema_learning/run_experiment.py"

sb_cmd = """#!/usr/bin/env bash
module load anaconda/4.4.0
source activate thesis
module load cudnn/cuda-9.0/7.0.3
module load cudatoolkit/9.0
"""

if args.function == "train":
    for trial_num in args.trial_nums:
        for model_name_and_num_epoch in args.model_names_and_num_epochs:
            output = "/home/cc27/Thesis/generalized_schema_learning/slurms/outputs/train_experiment{experiment_name}_{model_name}_{function}_trial{trial_num}_start{datetime}_{filler_type}%j.txt"
            cmd = sb_cmd + "\npython -u {script} --function={function} --exp_name={experiment_name} --filler_type={filler_type} --model_name={model_name} --trial_num={trial_num} --num_epochs={num_epochs}"
            model_name, num_epochs = model_name_and_num_epoch.split('_')
            this_output = output.format(model_name=model_name, function=args.function, experiment_name=args.experiment_name, datetime=time.strftime("%y%m%d") + "_" + time.strftime("%H%M%D").replace("/", ""), trial_num=trial_num, filler_type=args.filler_type)
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
            thiscmd = cmd.format(script=script, function=args.function, experiment_name=args.experiment_name, filler_type=args.filler_type, model_name=model_name, trial_num=trial_num, num_epochs=num_epochs)
            print(thiscmd)
            print('********')
            #print(this_output)
            sbproc.communicate(thiscmd)
elif args.function == "probe_ambiguous":
    for test_filename in args.test_filenames:
        for trial_num in args.trial_nums:
            for model_name in args.model_names:
                cmd = sb_cmd + "\npython -u {script} --function={function} --exp_name={experiment_name} --filler_type={filler_type} --model_name={model_name} --trial_num={trial_num} --test_filename={test_filename}"
                output = "/home/cc27/Thesis/generalized_schema_learning/slurms/outputs/probeambiguous_experiment{experiment_name}_{model_name}_{function}_trial{trial_num}_{datetime}_%j.txt"
                this_output = output.format(model_name=model_name, function=args.function, experiment_name=args.experiment_name, datetime=time.strftime("%y%m%d") + "_" + time.strftime("%H%M%D").replace("/", ""), trial_num=trial_num)
                print(this_output)
                sbproc = subprocess.Popen(["sbatch",
                                           "--output=" + this_output,
                                           "-J", model_name,
                                           "-p", "all",
                                           "-t", "1300",
                                           "--gres", "gpu:1",
                                           "--mail-type", "FAIL, BEGIN, END",
                                           "--mail-user", "cc27@alumni.princeton.edu"
                                          ],
                                      stdin=subprocess.PIPE)
                thiscmd = cmd.format(script=script, function=args.function, experiment_name=args.experiment_name, filler_type=args.filler_type, model_name=model_name, trial_num=trial_num, test_filename=test_filename)
                print(thiscmd)
                sbproc.communicate(thiscmd)
elif args.function == "test":
    for test_filename in args.test_filenames:
        for trial_num in args.trial_nums:
            for model_name in args.model_names:
                cmd = sb_cmd + "\npython -u {script} --function={function} --exp_name={experiment_name} --filler_type={filler_type} --model_name={model_name} --trial_num={trial_num} --test_filename={test_filename}"
                output = "/home/cc27/Thesis/generalized_schema_learning/slurms/outputs/probeambiguous_experiment{experiment_name}_{model_name}_{function}_trial{trial_num}_{datetime}_%j.txt"
                this_output = output.format(model_name=model_name, function=args.function, experiment_name=args.experiment_name, datetime=time.strftime("%y%m%d") + "_" + time.strftime("%H%M%D").replace("/", ""), trial_num=trial_num)
                print(this_output)
                sbproc = subprocess.Popen(["sbatch",
                                           "--output=" + this_output,
                                           "-J", model_name,
                                           "-p", "all",
                                           "-t", "1300",
                                           "--gres", "gpu:1",
                                           "--mail-type", "FAIL, BEGIN, END",
                                           "--mail-user", "cc27@alumni.princeton.edu"
                                          ],
                                      stdin=subprocess.PIPE)
                thiscmd = cmd.format(script=script, function=args.function, experiment_name=args.experiment_name, filler_type=args.filler_type, model_name=model_name, trial_num=trial_num, test_filename=test_filename)
                print(thiscmd)
                sbproc.communicate(thiscmd)
