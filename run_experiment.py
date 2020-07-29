"""Modules for contextual learning experiments using Coffee Shop World."""
import argparse
import directories
import embedding_util
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
import time

from data_util import generate_epoch
from architecture_models.model import fast_weights_model
from architecture_models.connect_DNC import dnc_model
from architecture_models.connect_NTM2 import ntm2_model
from architecture_models.custom_GRU import gru_model
from architecture_models.custom_LSTMLN import lstmln_model
from experiment_parameters import input_dims

base_dir = directories.base_dir

class parameters():
    def __init__(self):
        """FLAGS module.

        Contains experiment information and directory locations.
        """
        self.num_hidden_units = 50
        self.l = 0.95 # decay lambda
        self.e = 0.5 # learning rate eta
        self.S = 1 # num steps to get to h_S(t+1) (Parameter for Fast Weights.)
        self.learning_rate = 1e-4
        self.max_gradient_norm = 5.0

        self.data_dir = os.path.join(base_dir, 'data')
        self.results_dir = os.path.join(base_dir, 'results')

    def update_for_experiment(self, args):
        """Updates FLAGS parameters for specific experiment.

        Args:
            args: Parsed command line arguments.
        """
        DIMS = 50 # NOTE: hard-coded dimensions chosen for word vectors.
        self.filler_type = args.filler_type
        self.trial_num = args.trial_num
        self.experiment_name = args.exp_name
        self.model_name = args.model_name
        self.function = args.function
        self.input_dim = input_dims[self.experiment_name] # Length of input sequence, hard-coded in experiment_parameters.py.
        self.num_classes = DIMS # Num classes is vector dimension, since this is the size of vectors outputted by the networks.
        self.batch_size = 16
        self.data_dir = os.path.join(self.data_dir, self.experiment_name)
        if not args.checkpoint_filler_type:
            self.checkpoint_filler_type = self.filler_type
        else:
            self.checkpoint_filler_type = args.checkpoint_filler_type
        self.ckpt_dir = os.path.join(base_dir, 'checkpoints', self.experiment_name, self.checkpoint_filler_type, self.model_name, 'trial%d' % self.trial_num)
        self.results_dir = os.path.join(self.results_dir, self.experiment_name, self.checkpoint_filler_type, self.filler_type)

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

def get_embedding(FLAGS):
    """Retrieves word embedding for experiment.

    Args:
        FLAGS: Parameters for experiment.

    Returns:
        embedding_matrix: [num_words x embedding_dims] matrix of word embeddings.

    NOTE: Embeddings are created using experiment_creators/create_embedding.py.
    """
    with open(os.path.join(FLAGS.data_dir, 'embedding.p')) as f:
        embedding = pickle.load(f)
    embedding_dims = len(embedding[0]['vector'])
    num_words = len(embedding)
    embedding_matrix = np.empty([num_words, embedding_dims])
    for i in range(num_words):
        embedding_matrix[i,:] = embedding[i]['vector']
    return embedding_matrix

def get_clean_model(FLAGS):
    """Retrieves a previously untrained network.

    Args:
        FLAGS: Parameters for experiment, indicating model name.

    Returns:
        Randomly initialized network of specified model.

    Raises:
        ValueError: If requested architecture is not supported.
    """
    if FLAGS.model_name == 'GRU-LN':
        model = gru_model(FLAGS)
    elif FLAGS.model_name == 'LSTM-LN':
        model = lstmln_model(FLAGS)
    elif FLAGS.model_name == 'DNC':
        model = dnc_model(FLAGS)
    elif FLAGS.model_name == 'NTM2':
        model = ntm2_model(FLAGS)
    elif FLAGS.model_name in ['CONTROL', 'RNN-LN', 'RNN-LN-FW']:
        model = fast_weights_model(FLAGS)
    else:
        raise ValueError('Illegal model name')
    return model

def create_model(sess, FLAGS):
    """Creates model.

    If the architecture has been previously trained on this experiment, retrieves
    previously trained model. NTM2 refers to a reduced NTM; this model differs
    from a standard NTM in that it does not contain shift weights (i.e. it does
    not contain links between consecutive memory slots).

    Args:
        sess: Current tensorflow session.
        FLAGS: Parameters object for the experiment.

    Returns:
        model: Pretrained network if available, otherwise randomly initialized network.
        previous_train_epochs: Number of epochs this model has been trained (0 if untrained).

    Raises:
        ValueError: If requested architecture is not supported.
    """
    model = get_clean_model(FLAGS)
    ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
    if ckpt:
        latest_checkpoint_path = tf.train.latest_checkpoint(FLAGS.ckpt_dir)
        print("Restoring old model parameters from %s" % latest_checkpoint_path)
        model.saver.restore(sess, latest_checkpoint_path)
        previous_trained_epochs = int(latest_checkpoint_path.split("/")[-1].split(".")[-1].split("-")[-1]) # Better way to do this?
    else:
        print("Created new model.")
        sess.run(tf.global_variables_initializer())
        previous_trained_epochs = 0
    print("Previously trained for %d epochs" % previous_trained_epochs)
    return model, previous_trained_epochs

def load_data(data_path):
    """Helper module to load input and output data."""
    with open(data_path, 'rb') as f:
        X, y = pickle.load(f)
    print(data_path, "X shape: ", np.shape(X), "y shape", np.shape(y))
    X = np.array(X, dtype=int)
    y = np.array(y, dtype=int)
    return X, y

def get_meantestinfo(sess, test_X, test_y, FLAGS, model, embedding):
    """Helper module to get mean test accuracy and loss."""
    test_batch_accuracy = []
    test_batch_loss = []
    for test_epoch_num, test_epoch in enumerate(generate_epoch(test_X, test_y, num_epochs=1, FLAGS=FLAGS, embedding=embedding)):
        for test_batch_num, (batch_X, batch_y, batch_embedding) in enumerate(test_epoch):
            loss, accuracy = model.step(sess, batch_X, batch_y, batch_embedding, FLAGS.l, FLAGS.e, run_option="forward_only")
            test_batch_accuracy.append(accuracy)
            test_batch_loss.append(loss)
    return np.mean(test_batch_accuracy), np.mean(test_batch_loss)

def train(FLAGS):
    """Trains and tests the model.

    Note: experiments fall into two dataset types: Standard, and Curriculum Learning.
          Standard: Train data saved in train.p, test data saved in test.p.
          Curriculum Learning: Used for experiments that introduce new tasks
                               during training. Trainsets for current regime can
                               be manually tuned in the code. Currently hard-coded
                               for curricula with QSubject and QFriend.
    Note: Split datasets must be specifically created. See README/wiki for more details.
    Args:
        FLAGS: Parameters object for the experiment.

    Saves (saves information after every FLAGS.save_every train epochs):
        Checkpoints: Saves a checkpoint of the model state in the FLAGS.ckpt_dir folder.
        Results: Saves train and test accuracies in the FLAGS.results_dir folder.
        Split results: For certain experiments (hard-coded in this function), saves accuracies split by query type in the FLAGS.results_dir folder.
    """
    # Load the train/test datasets
    print("Loading datasets from directory %s:" % FLAGS.data_dir)
    if "curriculum" in FLAGS.experiment_name:
        train_X, train_y = load_data(os.path.join(FLAGS.data_dir, 'train_%s.p' % FLAGS.curriculum_regime))
    else:
        train_X, train_y = load_data(os.path.join(FLAGS.data_dir, 'train.p'))
    test_X, test_y = load_data(os.path.join(FLAGS.data_dir, 'test.p'))
    # Load split datasets.
    if 'QSubjectQFriend' in FLAGS.experiment_name or "curriculum" in FLAGS.experiment_name:
        test_X_subject, test_y_subject = load_data(os.path.join(FLAGS.data_dir, 'test_QSubject.p'))
        test_X_friend, test_y_friend = load_data(os.path.join(FLAGS.data_dir, 'test_QFriend.p'))
    elif 'AllQs' in FLAGS.experiment_name:
        test_X_subject, test_y_subject = load_data(os.path.join(FLAGS.data_dir, 'test_QSubject.p'))
        test_X_poet, test_y_poet = load_data(os.path.join(FLAGS.data_dir, 'test_QPoet.p'))
        test_X_dessert, test_y_dessert = load_data(os.path.join(FLAGS.data_dir, 'test_QDessert_bought.p'))
        test_X_drink, test_y_drink = load_data(os.path.join(FLAGS.data_dir, 'test_QDrink_bought.p'))
        test_X_emcee, test_y_emcee = load_data(os.path.join(FLAGS.data_dir, 'test_QEmcee.p'))
        test_X_friend, test_y_friend = load_data(os.path.join(FLAGS.data_dir, 'test_QFriend.p'))

    embedding = get_embedding(FLAGS)

    with tf.Session() as sess:
        # Load the model
        model, previous_trained_epochs = create_model(sess, FLAGS)
        start_time = time.time()

        # Start training. If applicable, load previous results.
        if previous_trained_epochs > 0:
            with open(os.path.join(FLAGS.results_dir, '%s_results_%depochs_trial%d.p' % (FLAGS.model_name, previous_trained_epochs, FLAGS.trial_num)), 'rb') as previous_results_file:
                train_epoch_accuracy, test_epoch_accuracy, train_epoch_loss, test_epoch_loss, train_epoch_gradient_norm = pickle.load(previous_results_file)
            if 'QSubjectQFriend' in FLAGS.experiment_name or "curriculum" in FLAGS.experiment_name:
                with open(os.path.join(FLAGS.results_dir, '%s_results_%depochs_trial%d_split.p' % (FLAGS.model_name, previous_trained_epochs, FLAGS.trial_num)), 'rb') as previous_results_file_split:
                    previous_results_split = pickle.load(previous_results_file_split)
                    test_epoch_accuracies_split = previous_results_split['accuracies']
                    test_epoch_losses_split = previous_results_split['losses']
            elif 'AllQs' in FLAGS.experiment_name:
                with open(os.path.join(FLAGS.results_dir, '%s_results_%depochs_trial%d_split.p' % (FLAGS.model_name, previous_trained_epochs, FLAGS.trial_num)), 'rb') as previous_results_file_split:
                    previous_results_split = pickle.load(previous_results_file_split)
                    test_epoch_accuracies_split, test_epoch_losses_split = previous_results_split['accuracies'], previous_results_split['losses']
        else:
                train_epoch_loss = []; test_epoch_loss = []
                train_epoch_accuracy = []; test_epoch_accuracy = []
                train_epoch_gradient_norm = []
                if 'QSubjectQFriend' in FLAGS.experiment_name or "curriculum" in FLAGS.experiment_name:
                    test_epoch_accuracies_split = {'QFriend':[], 'QSubject':[]}
                    test_epoch_losses_split = {'QFriend':[], 'QSubject':[]}
                elif 'AllQs' in  FLAGS.experiment_name:
                    test_epoch_accuracies_split = {'QDrink':[], 'QDessert':[], 'QEmcee':[], 'QFriend':[], 'QPoet':[], 'QSubject':[]}
                    test_epoch_losses_split = {'QDrink':[], 'QDessert':[], 'QEmcee':[], 'QFriend':[], 'QPoet':[], 'QSubject':[]}
        for train_epoch_num, train_epoch in enumerate(generate_epoch(train_X, train_y, FLAGS.num_epochs, FLAGS, embedding)):
            test_start_time = time.time()
            train_epoch_num += 1 # Use 1-based indexing for train epoch numbering.
            print("EPOCH:", train_epoch_num + previous_trained_epochs)
            sess.run(tf.assign(model.lr, FLAGS.learning_rate)) # Assign the learning rate.

            # Train.
            train_batch_loss = []
            train_batch_accuracy = []
            train_batch_gradient_norm = []
            for train_batch_num, (batch_X, batch_y, batch_embedding) in enumerate(train_epoch):
                loss, accuracy, norm, _ = model.step(sess, batch_X, batch_y, batch_embedding, FLAGS.l, FLAGS.e, run_option="backprop")
                train_batch_loss.append(loss)
                train_batch_accuracy.append(accuracy)
                train_batch_gradient_norm.append(norm)
            train_epoch_loss.append(np.mean(train_batch_loss))
            train_epoch_accuracy.append(np.mean(train_batch_accuracy))
            train_epoch_gradient_norm.append(np.mean(train_batch_gradient_norm))
            print ('Epoch: [%i/%i] time: %.4f, loss: %.7f,'
                    ' acc: %.7f, norm: %.7f' % (train_epoch_num, FLAGS.num_epochs,
                        time.time() - start_time, train_epoch_loss[-1],
                        train_epoch_accuracy[-1], train_epoch_gradient_norm[-1]))

            # Test.
            test_start_time = time.time()
            test_batch_loss = []
            test_batch_accuracy = []
            mean_test_batch_accuracy, mean_test_batch_loss = get_meantestinfo(sess, test_X, test_y, FLAGS, model, embedding)
            test_epoch_accuracy.append(mean_test_batch_accuracy)
            test_epoch_loss.append(mean_test_batch_loss)

            if 'QSubjectQFriend' in FLAGS.experiment_name  or "curriculum" in FLAGS.experiment_name:
                mean_test_batch_accuracy_friend, mean_test_batch_loss_friend = get_meantestinfo(sess, test_X_friend, test_y_friend, FLAGS, model, embedding)
                mean_test_batch_accuracy_subject, mean_test_batch_loss_subject = get_meantestinfo(sess, test_X_subject, test_y_subject, FLAGS, model, embedding)
                test_epoch_accuracies_split['QFriend'].append(mean_test_batch_accuracy_friend)
                test_epoch_losses_split['QFriend'].append(mean_test_batch_loss_friend)
                test_epoch_accuracies_split['QSubject'].append(mean_test_batch_accuracy_subject)
                test_epoch_losses_split['QSubject'].append(mean_test_batch_loss_subject)
            elif 'AllQs' in FLAGS.experiment_name:
                mean_test_batch_accuracy_dessert, mean_test_batch_loss_dessert = get_meantestinfo(sess, test_X_dessert, test_y_dessert, FLAGS, model, embedding)
                mean_test_batch_accuracy_drink, mean_test_batch_loss_drink = get_meantestinfo(sess, test_X_drink, test_y_drink, FLAGS, model, embedding)
                mean_test_batch_accuracy_emcee, mean_test_batch_loss_emcee = get_meantestinfo(sess, test_X_emcee, test_y_emcee, FLAGS, model, embedding)
                mean_test_batch_accuracy_friend, mean_test_batch_loss_friend = get_meantestinfo(sess, test_X_friend, test_y_friend, FLAGS, model, embedding)
                mean_test_batch_accuracy_poet, mean_test_batch_loss_poet = get_meantestinfo(sess, test_X_poet, test_y_poet, FLAGS, model, embedding)
                mean_test_batch_accuracy_subject, mean_test_batch_loss_subject = get_meantestinfo(sess, test_X_subject, test_y_subject, FLAGS, model, embedding)
                test_epoch_accuracies_split['QDessert'].append(mean_test_batch_accuracy_dessert)
                test_epoch_losses_split['QDessert'].append(mean_test_batch_loss_dessert)
                test_epoch_accuracies_split['QDrink'].append(mean_test_batch_accuracy_drink)
                test_epoch_losses_split['QDrink'].append(mean_test_batch_loss_drink)
                test_epoch_accuracies_split['QEmcee'].append(mean_test_batch_accuracy_emcee)
                test_epoch_losses_split['QEmcee'].append(mean_test_batch_loss_emcee)
                test_epoch_accuracies_split['QFriend'].append(mean_test_batch_accuracy_friend)
                test_epoch_losses_split['QFriend'].append(mean_test_batch_loss_friend)
                test_epoch_accuracies_split['QPoet'].append(mean_test_batch_accuracy_poet)
                test_epoch_losses_split['QPoet'].append(mean_test_batch_loss_poet)
                test_epoch_accuracies_split['QSubject'].append(mean_test_batch_accuracy_subject)
                test_epoch_losses_split['QSubject'].append(mean_test_batch_loss_subject)
            print ('Epoch: [%i/%i] time: %.4f, test loss: %.7f,'
                    ' test acc: %.7f' % (train_epoch_num, FLAGS.num_epochs,
                        time.time() - test_start_time, test_epoch_loss[-1],
                        test_epoch_accuracy[-1]))

            # Save model and dump results.
            if (train_epoch_num % FLAGS.save_every == 0 or train_epoch_num == (FLAGS.num_epochs)) and (train_epoch_num > 0):
                if not os.path.isdir(FLAGS.ckpt_dir):
                    os.makedirs(FLAGS.ckpt_dir)
                checkpoint_path = os.path.join(FLAGS.ckpt_dir, "%s" % (FLAGS.model_name))
                print("Saving the model and results at epoch %d." % (train_epoch_num + previous_trained_epochs))
                model.saver.save(sess, checkpoint_path, global_step=(train_epoch_num + previous_trained_epochs))
                with open(os.path.join(FLAGS.results_dir, '%s_results_%depochs_trial%d.p' % (FLAGS.model_name, train_epoch_num + previous_trained_epochs, FLAGS.trial_num)), 'wb') as f:
                    pickle.dump([train_epoch_accuracy, test_epoch_accuracy, train_epoch_loss, test_epoch_loss, train_epoch_gradient_norm], f)
                if 'QSubjectQFriend' in FLAGS.experiment_name or "curriculum" in FLAGS.experiment_name:
                    with open(os.path.join(FLAGS.results_dir, '%s_results_%depochs_trial%d_split.p' % (FLAGS.model_name, train_epoch_num + previous_trained_epochs, FLAGS.trial_num)), 'wb') as f:
                        pickle.dump({'accuracies':test_epoch_accuracies_split, 'losses':test_epoch_losses_split}, f)
                elif 'AllQs' in FLAGS.experiment_name:
                    with open(os.path.join(FLAGS.results_dir, '%s_results_%depochs_trial%d_split.p' % (FLAGS.model_name, train_epoch_num + previous_trained_epochs, FLAGS.trial_num)), 'wb') as f:
                        pickle.dump({'accuracies':test_epoch_accuracies_split, 'losses':test_epoch_losses_split}, f)

def test(FLAGS, test_filename, save_logits=False):
    """Perform error analysis on test set.

    Args:
        FLAGS: Parameters object for the experiment.
        test_filename: Name of file containing desired test set.

    Saves:
        Error analysis results: Saves test inputs, predictions, and responses in
                                the FLAGS.results_dir/predictions/ folder.
    """
    # Load the train/test datasets
    print("Testing")
    print("Loading datasets from directory %s:" % FLAGS.data_dir)
    test_X, test_y = load_data(os.path.join(FLAGS.data_dir, test_filename))
    embedding = get_embedding(FLAGS)
    with tf.Session() as sess:
        # Load the model.
        model, previous_trained_epochs = create_model(sess, FLAGS)
        start_time = time.time()

        # Test.
        test_start_time = time.time()
        test_batch_loss = []
        test_batch_accuracy = []
        inputs = []
        predictions = []
        responses = []
        saved_logits = []
        for test_epoch_num, test_epoch in enumerate(generate_epoch(test_X, test_y, num_epochs=1, FLAGS=FLAGS, embedding=embedding, do_shift_inputs=False)):
            for test_batch_num, (batch_X, batch_y, batch_embedding) in enumerate(test_epoch):
                print(test_batch_num)
                if test_batch_num > 5:
                    break
                loss, accuracy = model.step(sess, batch_X, batch_y, batch_embedding, FLAGS.l, FLAGS.e, run_option="forward_only")
                test_batch_loss.append(loss)
                test_batch_accuracy.append(accuracy)
                logits = model.logits.eval(feed_dict={model.X: batch_X, model.embedding: batch_embedding, model.l: FLAGS.l, model.e: FLAGS.e})
                for i in range(FLAGS.batch_size):
                    test_input = [embedding_util.get_corpus_index(vector, batch_embedding) for vector in batch_X[i]]
                    prediction = embedding_util.get_corpus_index(logits[i], batch_embedding)
                    response = embedding_util.get_corpus_index(batch_y[i], batch_embedding)
                    inputs.append(test_input)
                    predictions.append(prediction)
                    responses.append(response)
                    if save_logits:
                        saved_logits.append(np.expand_dims(logits[i], axis=0))
                print('test batch accuracy %0.2f' % np.mean(test_batch_accuracy))
        print('Test set name %s' % str(test_filename))
        print ('Test time: %.4f, test loss: %.7f,'
            ' test acc: %.7f' % (time.time() - test_start_time, np.mean(test_batch_loss),
            np.mean(test_batch_accuracy)))
        analysis_results = pd.DataFrame(
                        {'inputs':inputs,
                        'predictions':predictions,
                        'responses':responses
                        })
        predictions_dir = os.path.join(FLAGS.results_dir, 'predictions')
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)
        with open(os.path.join(predictions_dir, 'test_analysis_results_%s_%depochs_trial%d_%s' % (FLAGS.model_name, previous_trained_epochs, FLAGS.trial_num, test_filename)), 'wb') as f:
            pickle.dump(analysis_results, f)
        if save_logits:
            np.savez(os.path.join(predictions_dir, 'logits_%s_%depochs_trial%d_%s' % (FLAGS.model_name, previous_trained_epochs, FLAGS.trial_num, test_filename)), np.concatenate(saved_logits, axis=0))


def analyze(FLAGS, test_filename):
   """Perform analysis on specified test set. Used for decoding experiments.

   NOTE: Currently works for select models: NTM2, RNN-LN-FW, LSTM-LN.
   Args:
       FLAGS: Parameters object for the experiment.
       test_filename: Name of file containing desired test set.

   Saves:
        Experiment inputs, correct outputs, network predictions, and network state histories.
   """
   # Load the train/test datasets
   if FLAGS.model_name not in ["NTM2", "LSTM-LN", "RNN-LN-FW", "RNN-LN"]:
       raise ArgumentError("Analysis currently works only with RNN, LSTM, Fast Weights, and reduced NTM.")
   print("Running controller and external memory buffer analysis")
   print("Loading datasets from directory %s:" % FLAGS.data_dir)
   test_X, test_y = load_data(os.path.join(FLAGS.data_dir, test_filename))
   embedding = get_embedding(FLAGS)
   with tf.Session() as sess:

       # Load the model
       model, previous_trained_epochs = create_model(sess, FLAGS)
       start_time = time.time()

       # test set
       test_start_time = time.time()
       test_batch_loss = []
       test_batch_accuracy = []
       inputs = []
       predictions = []
       responses = []
       first_epoch = True
       for test_epoch_num, test_epoch in enumerate(generate_epoch(test_X, test_y, num_epochs=1, FLAGS=FLAGS, embedding=embedding)):
           for test_batch_num, (batch_X, batch_y, batch_embedding) in enumerate(test_epoch):
               print(test_batch_num)
               loss, accuracy, controller_history, memory_history = model.step(sess, batch_X, batch_y, batch_embedding, FLAGS.l, FLAGS.e, run_option="analyze")
               test_batch_loss.append(loss)
               test_batch_accuracy.append(accuracy)
               if first_epoch:
                   first_epoch = False
                   gate_histories = controller_history[0]
                   hidden_histories = controller_history[1]
                   memory_histories = memory_history
                   output_vectors = batch_y
                   input_vectors = batch_X
                   batch_embeddings = batch_embedding
               else:
                   if FLAGS.model_name in ["NTM2", "LSTM-LN"]:
                       gate_histories = np.concatenate((gate_histories, controller_history[0]), axis=0)
                   hidden_histories = np.concatenate((hidden_histories, controller_history[1]), axis=0)
                   if FLAGS.model_name in ["NTM2", "RNN-LN-FW"]:
                       memory_histories = np.concatenate((memory_histories, memory_history), axis=0)
                   output_vectors = np.concatenate((output_vectors, batch_y), axis=0)
                   input_vectors = np.concatenate((input_vectors, batch_X), axis=0)
                   batch_embeddings = np.concatenate((batch_embeddings, batch_embedding), axis=0)
               logits = model.logits.eval(feed_dict={model.X: batch_X, model.embedding: batch_embedding,
                   model.l: FLAGS.l, model.e: FLAGS.e})
               for i in range(FLAGS.batch_size):
                   test_input = [embedding_util.get_corpus_index(vector, batch_embedding) for vector in batch_X[i]]
                   prediction = embedding_util.get_corpus_index(logits[i], batch_embedding)
                   response = embedding_util.get_corpus_index(batch_y[i], batch_embedding)
                   inputs.append(test_input)
                   predictions.append(prediction)
                   responses.append(response)
       print('Test set name %s' % str(test_filename))
       print ('Test time: %.4f, test loss: %.7f,'
           ' test acc: %.7f' % (time.time() - test_start_time, np.mean(test_batch_loss),
           np.mean(test_batch_accuracy)))
       analysis_results = pd.DataFrame(
                       {'inputs':inputs,
                       'predictions':predictions,
                       'responses':responses
                       })
       predictions_dir = os.path.join(FLAGS.results_dir, 'analysis')
       if not os.path.exists(predictions_dir):
           os.makedirs(predictions_dir)
       with open(os.path.join(predictions_dir, 'test_analysis_results_%s_%depochs_trial%d_%s' % (FLAGS.model_name, previous_trained_epochs, FLAGS.trial_num, test_filename)), 'wb') as f:
           pickle.dump(analysis_results, f)
       with open(os.path.join(predictions_dir, 'batch_embeddings_%s_%depochs_trial%d_%s' % (FLAGS.model_name, previous_trained_epochs, FLAGS.trial_num, test_filename.replace(".p",".npz"))), 'wb') as f:
           np.savez(f, batch_embeddings)
       with open(os.path.join(predictions_dir, 'input_vectors_%s_%depochs_trial%d_%s' % (FLAGS.model_name, previous_trained_epochs, FLAGS.trial_num, test_filename.replace(".p",".npz"))), 'wb') as f:
           np.savez(f, input_vectors)
       with open(os.path.join(predictions_dir, 'output_vectors_%s_%depochs_trial%d_%s' % (FLAGS.model_name, previous_trained_epochs, FLAGS.trial_num, test_filename.replace(".p",".npz"))), 'wb') as f:
           np.savez(f, output_vectors)
       with open(os.path.join(predictions_dir, 'hidden_histories_%s_%depochs_trial%d_%s' % (FLAGS.model_name, previous_trained_epochs, FLAGS.trial_num, test_filename.replace(".p",".npz"))), 'wb') as f:
           np.savez(f, hidden_histories)
       if FLAGS.model_name in ["NTM2", "LSTM-LN"]:
           with open(os.path.join(predictions_dir, 'gate_histories_%s_%depochs_trial%d_%s' % (FLAGS.model_name, previous_trained_epochs, FLAGS.trial_num, test_filename.replace(".p",".npz"))), 'wb') as f:
               np.savez(f, gate_histories)
       if FLAGS.model_name in ["NTM2", "RNN-LN-FW"]:
           with open(os.path.join(predictions_dir, 'memory_histories_%s_%depochs_trial%d_%s' % (FLAGS.model_name, previous_trained_epochs, FLAGS.trial_num, test_filename.replace(".p",".npz"))), 'wb') as f:
               np.savez(f, memory_histories)

if __name__ == '__main__':

    FLAGS = parameters()
    parser=argparse.ArgumentParser()

    parser.add_argument('--function', help='Desired function.', choices=["train", "test", "analyze", "probe"], required=True)
    parser.add_argument('--exp_name', help='Name of folder containing experiment data.', type=str, required=True)
    parser.add_argument('--filler_type', help='Filler representation method', choices=["fixed_filler", "variable_filler", "variable_filler_distributions"], required=True)
    parser.add_argument('--checkpoint_filler_type', help='Filler representation method', choices=["fixed_filler", "variable_filler", "variable_filler_distributions"], required=True)
    parser.add_argument('--model_name', help='Name of architecture.', choices=["CONTROL", "DNC", "GRU-LN", "LSTM-LN", "NTM2", "RNN-LN", "RNN-LN-FW"], required=True)

    parser.add_argument('--num_epochs', help='Number of epochs to train. Only used for train function.', type=int)

    parser.add_argument('--test_filename', help='Required if using test function: Name of file containing test data.', type=str)

    parser.add_argument('--regime', help='Required if running curriculum experiment.', choices=["SUBJECT", "POET", "COMBINED"])
    parser.add_argument('--trial_num', help='Integer label for trial.', type=int, required=True)
    args=parser.parse_args()

    # Choose experiment.
    FLAGS.update_for_experiment(args)

    print("*******************************************************************")
    print("Experiment: %s" % FLAGS.experiment_name)
    print("Model: %s" % (FLAGS.model_name))
    print("Filler type: %s" % (FLAGS.filler_type))
    print("Data directory: %s" % FLAGS.data_dir)
    print("Checkpoint directory: %s" % FLAGS.ckpt_dir)
    print("Results directory: %s" % FLAGS.results_dir)
    print("*******************************************************************")

    np.random.seed(args.trial_num)
    tf.set_random_seed(args.trial_num)
    if args.function == 'train':
        FLAGS.num_epochs = int(args.num_epochs)
        FLAGS.save_every =  max(1, FLAGS.num_epochs//4)
        if "curriculum" in FLAGS.experiment_name:
            FLAGS.curriculum_regime = args.regime
        train(FLAGS)
    elif args.function == 'test':
        test_filename = args.test_filename
        test(FLAGS, test_filename)
    elif args.function == 'probe':
        test_filename = args.test_filename
        test(FLAGS, test_filename, save_logits=True)
    elif args.function == 'analyze':
        test_filename = args.test_filename
        analyze(FLAGS, test_filename)
