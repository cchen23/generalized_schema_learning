import argparse
import subprocess
import time

# Info: https://slurm.schedmd.com/sbatch.html
script = "/home/cc27/Thesis/generalized_schema_learning/analysis/decode_history.py"

sb_cmd = """#!/usr/bin/env bash
module load anaconda/4.4.0
source activate thesis
module load cudnn/cuda-9.0/7.0.3
module load cudatoolkit/9.0
"""
for model in ['RNN-LN-FW', 'RNN-LN', 'LSTM-LN', 'NTM2']:
    for filler in ['Subject', 'Friend', 'Emcee', 'Drink', 'Poet', 'Dessert']:
        output = "/home/cc27/Thesis/generalized_schema_learning/slurms/outputs/decode_{filler}_{model}%j.txt"
        cmd = sb_cmd + "\npython -u {script} AllQs 30000 {filler} {model}"
        this_output = output.format(filler=filler, model=model)
        sbproc = subprocess.Popen(["sbatch",
                                   "--output=" + this_output,
                                   "-J", model,
                                   "-p", "all",
                                   "-t", "100",
                                   "--mail-type", "FAIL,BEGIN,END",
                                   "--mail-user", "cc27@alumni.princeton.edu"
                                  ],
                              stdin=subprocess.PIPE)
        thiscmd = cmd.format(script=script, filler=filler, model=model)
        print(thiscmd)
        print('********')
        #print(this_output)
        sbproc.communicate(thiscmd)
