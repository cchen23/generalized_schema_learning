import os

os.system('python run_on_cluster.py --experiment-name variablefiller_AllQs --filler-type variable_filler_distributions_5050_AB_noise --model-names RNN-LN-FW NTM2-xl --test-filenames test_QDESSERT_double.p test_QDRINK.p test_QEMCEE_double.p test_QFRIEND.p test_QPOET.p test_QSUBJECT.p --trial-nums 40 41 42 43 44 --function probe_ambiguous --checkpoint-filler-type variable_filler_distributions')

#os.system('python run_on_cluster.py --experiment-name variablefiller_AllQs --filler-type variable_filler_distributions_5050_AB --model-names RNN-LN-FW NTM2-xl --test-filenames test_QFRIEND.p --trial-nums 40 41 42 43 44 --function probe_ambiguous --checkpoint-filler-type variable_filler_distributions_second_order_subject')

#os.system('python run_on_cluster.py --experiment-name variablefiller_AllQs --filler-type variable_filler_distributions_fixed_subject --model-names RNN-LN-FW NTM2-xl --test-filenames test_QFRIEND.p --trial-nums 40 41 42 43 44 --function probe_ambiguous --checkpoint-filler-type variable_filler_distributions_fixed_subject')

#filler_types = ['variable_filler_distributions_5050_AB', 'variable_filler_distributions_A', 'variable_filler_distributions_B', 'variable_filler_distributions_all_randn_distribution']
#checkpoint_types = ['variable_filler_distributions']#, 'variable_filler_distributions_second_order_subject', 'variable_filler_distributions_fixed_subject']
#
#for filler_type in filler_types:
#    for checkpoint_type in checkpoint_types:
#        #os.system('python run_on_cluster.py --experiment-name variablefiller_AllQs --filler-type {filler_type} --model-names RNN-LN-FW NTM2-xl --trial-nums 40 41 42 43 44 45 --function test --test-filenames test_QDESSERT_double.p test_QDRINK.p test_QEMCEE_double.p test_QFRIEND.p test_QPOET.p test_QSUBJECT.p --checkpoint-filler-type {checkpoint_filler_type}'.format(filler_type=filler_type, checkpoint_filler_type=checkpoint_type))
#        os.system('python run_on_cluster.py --experiment-name variablefiller_AllQs --filler-type {filler_type} --model-names RNN-LN-FW --trial-nums 41 --function test --test-filenames test_QDESSERT_double.p test_QDRINK.p test_QEMCEE_double.p test_QFRIEND.p test_QPOET.p test_QSUBJECT.p --checkpoint-filler-type {checkpoint_filler_type}'.format(filler_type=filler_type, checkpoint_filler_type=checkpoint_type))
