"""Modules to generate train epochs."""
import numpy as np
import embedding_util

def generate_epoch(X, y, num_epochs, FLAGS, embedding, do_shift_inputs=True):
    """Generate a train epoch.

    Args:
        X: [num_examples x num_words_per_input x 1] matrix of inputs.
        y: [num_examples x 1] matrix of correct outputs.
        num_epochs: Number of epochs to generate.
        FLAGS: Parameters object of experiment information.
        embedding: [num_words x embedding_dims] matrix of word embeddings.
                   NOTE: irrelevant if using one-hot embedding.
    Returns:
        A generator containing num_epoch batches.
    """
    for epoch_num in range(num_epochs):
        yield generate_batch(X, y, FLAGS, embedding, do_shift_inputs=do_shift_inputs)

def generate_batch(X, y, FLAGS, embedding, do_shift_inputs=True):
    """Generate a train batch.

    Constructs batches using one of three possible representations (specified by
    FLAGS.filler_type):
        fixed_filler: Each word vector is specified by the embedding argument.
        variable_filler: Each non-filler word vector is
                                          specified by the embedding argument.
                                          Each filler word (manually specified
                                          for each experiment) represented by a
                                          new randomly generated vector in each
                                          story.
    Args:
        X: [num_examples x num_words_per_input x 1] matrix of inputs.
        y: [num_examples x 1] matrix of correct outputs.
        FLAGS: Parameters object of experiment information.
        embedding: [num_words x embedding_dims] matrix of word embeddings.
                   NOTE: irrelevant if using one-hot embedding.
    Returns:
        A generator containing batch_size examples, each of which contains:
            X: [batch_size x num_words_per_input x num_dimensions_per_word] matrix
               of inputs.
            y: [batch_size x num_dimensions_per_word] matrix of correct outputs.
            embedding: [num_words_in_corpus x num_dimensions_per_word] matrix
                       of vectors representing words in the batch.
    """
    num_classes, batch_size, filler_type = FLAGS.num_classes, FLAGS.batch_size, FLAGS.filler_type
    data_size = len(X)
    num_batches = (data_size // batch_size)
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if filler_type == "fixed_filler":
            if do_shift_inputs:
                yield embedding[shift_inputs(X[start_index:end_index].squeeze(), FLAGS.experiment_name)], embedding[y[start_index:end_index].squeeze()], embedding
            else:
                yield embedding[X[start_index:end_index]].squeeze(), embedding[y[start_index:end_index].squeeze()], embedding
        elif filler_type == "variable_filler_distributions":
            DIMS = 50
            batchX, batchy = X[start_index:end_index].squeeze(), y[start_index:end_index].squeeze()
            if do_shift_inputs:
                batchX, padding_location = shift_inputs(batchX, FLAGS.experiment_name, return_padding_location=True)
            else:
                padding_location = np.inf
            embeddingX, embeddingy = embedding[batchX], embedding[batchy]
            epoch_embedding = embedding
            if FLAGS.experiment_name == 'probe_role_statistic_recall_normalize_90':
                FILLER_DISTRIBUTIONS = [{'distribution': 'A', 'indices': [3]}, 
                        {'distribution': 'B', 'indices': [4]},
                        {'distribution': 'B', 'indices': [6]},
                        {'distribution': 'A', 'indices': [8]}]
                for example_num in range(batch_size):
                    num_fillers = len(FILLER_DISTRIBUTIONS)
                    new_filler_embedding = np.empty((num_fillers, DIMS))
                    for j, distribution_info in enumerate(FILLER_DISTRIBUTIONS):
                        sample_value = np.random.rand()
                        distribution = distribution_info['distribution']
                        if (distribution == 'A' and sample_value > 0.9) or (distribution == 'B' and sample_value < 0.1):
                            filler_distribution = 'add05even'
                        else:
                            filler_distribution = 'add05odd'
                        new_filler_vector = embedding_util.create_word_vector(filler_distribution=filler_distribution)
                        new_filler_embedding[j,:] = new_filler_vector
                        filler_indices = distribution_info['indices']
                        # Replace filler embedding with new random embedding.
                        for filler_index in filler_indices:
                            if filler_index >= padding_location:
                                actual_filler_index = filler_index + 1
                            else:
                                actual_filler_index = filler_index
                            if (embeddingy[example_num] == embeddingX[example_num, actual_filler_index]).all():
                                embeddingy[example_num] = new_filler_vector
                            embeddingX[example_num, actual_filler_index] = new_filler_vector
                    # Append embedding to original embedding identifying response.
                    epoch_embedding = np.concatenate((epoch_embedding, new_filler_embedding), axis=0)
                yield embeddingX, embeddingy, epoch_embedding

        elif filler_type == "variable_filler":
            # NOTE: Filler indices manually determined using word list saved by experiment creators.
            if FLAGS.experiment_name == "variablefiller_gensymbolicstates_100000_1_testunseen_QSubject":
                FILLER_INDICES = [0, 5, 6, 13, 17, 18, 19, 20, 21, 22, 23, 24]
            elif FLAGS.experiment_name == "variablefiller_gensymbolicstates_100000_1_testunseen_AllQs":
                FILLER_INDICES = [0, 6, 7, 14, 19, 20, 24, 25, 26, 27, 28, 29]
            else:
                raise ValueError("Unsupported experiment name")
            DIMS = 50
            batchX, batchy = X[start_index:end_index].squeeze(), y[start_index:end_index].squeeze()
            if FLAGS.function != "analyze" and do_shift_inputs: # Don't randomly shift inputs for decoding analysis.
                batchX = shift_inputs(batchX, FLAGS.experiment_name)
            embeddingX, embeddingy = embedding[batchX], embedding[batchy]
            epoch_embedding = embedding
            for examplenum in range(batch_size):
                # Create new random embedding for each filler.
                num_fillers = len(FILLER_INDICES)
                new_filler_embedding = np.empty((num_fillers, DIMS))
                for j in range(num_fillers):
                    new_filler_embedding[j,:] = embedding_util.create_word_vector()
                # Replace filler embedding with new random embedding.
                filler_ix_X = np.where(np.isin(batchX[examplenum], FILLER_INDICES))
                new_embedding_ix_X = [FILLER_INDICES.index(i) for i in batchX[examplenum,filler_ix_X][0]]
                embeddingX[examplenum,filler_ix_X] = new_filler_embedding[new_embedding_ix_X]
                new_embedding_ix_y = [FILLER_INDICES.index(batchy[examplenum])]
                embeddingy[examplenum] = new_filler_embedding[new_embedding_ix_y]
                # Append embedding to original embedding identifying response.
                epoch_embedding = np.concatenate((epoch_embedding, new_filler_embedding), axis=0)
            yield embeddingX, embeddingy, epoch_embedding

def shift_inputs(batchX, experiment_name, return_padding_location=False):
    # NOTE: Padding indices manually determined using word list saved by experiment creators.
    if experiment_name == "variablefiller_gensymbolicstates_100000_1_testunseen_QSubject":
        padding_index = 15
    elif experiment_name == "variablefiller_gensymbolicstates_100000_1_testunseen_AllQs":
        padding_index = 16
    elif experiment_name == "fixedfiller_gensymbolicstates_100000_1_AllQs":
        padding_index = 27
    elif experiment_name == "fixedfiller_gensymbolicstates_100000_1_testunseen_AllQs":
        padding_index = 27
    elif experiment_name in ["probe_role_statistic_recall", "probe_role_statistic_recall_normalize"]:
        padding_index = 5017
    elif experiment_name in ["probe_role_statistic_recall_normalize_75", "probe_role_statistic_recall_normalize_90", "probe_role_statistic_recall_normalize_100"]:
        padding_index = 23017
    else:
        raise Exception("Unsupported experiment name.")
    batch_size = batchX.shape[0]
    new_X = np.zeros(batchX.shape, dtype=np.int16)
    for i in range(len(batchX)):
        original_Xi = batchX[i]
        new_Xi = np.zeros(original_Xi.shape, dtype=int)
        first_padding_index = np.where(original_Xi == padding_index)[0][0]
        padding_location = np.random.choice(range(first_padding_index))
        new_Xi[:padding_location] = original_Xi[:padding_location]
        new_Xi[padding_location] = padding_index
        new_Xi[padding_location+1:first_padding_index+1] = original_Xi[padding_location:first_padding_index]
        new_Xi[first_padding_index+1:] = original_Xi[first_padding_index+1:]
        new_X[i] = new_Xi
    if return_padding_location:
        return new_X, padding_location
    else:
        return new_X
