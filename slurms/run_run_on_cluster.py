import os

os.system('python run_on_cluster.py --experiment-name variablefiller_AllQs --filler-type variable_filler_distributions_5050_AB_noise --model-names RNN-LN-FW NTM2-xl --test-filenames test_QDESSERT_double.p test_QDRINK.p test_QEMCEE_double.p test_QFRIEND.p test_QPOET.p test_QSUBJECT.p --trial-nums 40 41 42 43 44 --function probe_ambiguous --checkpoint-filler-type variable_filler_distributions')
