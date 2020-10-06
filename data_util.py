"""Modules to generate train epochs."""
import numpy as np
import embedding_util

from hard_coded_things import experiment_parameters, embedding_size

def generate_epoch(X, y, num_epochs, FLAGS, embedding, do_shift_inputs=True, noise_proportion=0.1, zero_vector_noise=False):
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
        yield generate_batch(X, y, FLAGS, embedding, do_shift_inputs=do_shift_inputs, noise_proportion=noise_proportion, zero_vector_noise=zero_vector_noise)

def generate_batch(X, y, FLAGS, embedding, do_shift_inputs=True, noise_proportion=0.1, zero_vector_noise=False):
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
        elif "variable_filler" in filler_type:
            # NOTE: Filler indices manually determined using word list saved by experiment creators.
            if "distributions" in filler_type:
                filler_indices_and_distributions = experiment_parameters["filler_distributions"][FLAGS.experiment_name]
                query_to_filler_indices = experiment_parameters["query_to_filler_index"][FLAGS.experiment_name]
                filler_indices = list(filler_indices_and_distributions.keys())
                filler_distributions = [filler_indices_and_distributions[filler_index] for filler_index in filler_indices]
                filler_indices = [int(index) for index in filler_indices]
            else:
                filler_indices = experiment_parameters["filler_indices"][FLAGS.experiment_name]
            batchX, batchy = X[start_index:end_index].squeeze(), y[start_index:end_index].squeeze()
            if FLAGS.function != "analyze" and do_shift_inputs: # Don't randomly shift inputs for decoding analysis.
                batchX = shift_inputs(batchX, FLAGS.experiment_name)
            embeddingX, embeddingy = embedding[batchX], embedding[batchy]
            padding_index = experiment_parameters['padding_indices'][FLAGS.experiment_name]
            padding_vector = embedding[padding_index]
            epoch_embedding = embedding
            for examplenum in range(batch_size):
                # Create new random embedding for each filler.
                num_fillers = len(filler_indices)
                new_filler_embedding = np.empty((num_fillers, embedding_size))
                if "distributions" in filler_type:
                    subject_distribution = None
                    if "second_order_subject" in filler_type:
                        indices_to_fillers = {1: 'EMCEEFILLER', 12: 'SUBJECTFILLER', 19: 'DESSERTFILLER', 21: 'DRINKFILLER', 23: 'POETFILLER', 27: 'FRIENDFILLER'}
                        subject_distribution = np.random.choice(['A', 'B'])
                        for j, filler_index in enumerate(filler_indices):
                            if indices_to_fillers[filler_index] == "SUBJECTFILLER":
                                new_filler_embedding[j,:] = embedding_util.create_word_vector(filler_distribution=subject_distribution, dominant_distribution_proportion=1)
                            elif indices_to_fillers[filler_index] in ["FRIENDFILLER"]:
                                new_filler_embedding[j,:] = embedding_util.create_word_vector(filler_distribution=subject_distribution, dominant_distribution_proportion=0.9)
                            else:
                                new_filler_embedding[j,:] = embedding_util.create_word_vector(filler_distribution="A", dominant_distribution_proportion=0.5)
                    elif "fixed_subject" in filler_type:
                        indices_to_fillers = {1: 'EMCEEFILLER', 12: 'SUBJECTFILLER', 19: 'DESSERTFILLER', 21: 'DRINKFILLER', 23: 'POETFILLER', 27: 'FRIENDFILLER'}
                        subject_distribution = np.random.choice(['A_fixed', 'B_fixed'])
                        for j, filler_index in enumerate(filler_indices):
                            if indices_to_fillers[filler_index] == "SUBJECTFILLER":
                                new_filler_embedding[j,:] = embedding_util.create_word_vector(filler_distribution=subject_distribution, dominant_distribution_proportion=1)
                            elif indices_to_fillers[filler_index] in ["FRIENDFILLER"]:
                                new_filler_embedding[j,:] = embedding_util.create_word_vector(filler_distribution=subject_distribution + '_filler', dominant_distribution_proportion=0.9)
                            else:
                                new_filler_embedding[j,:] = embedding_util.create_word_vector(filler_distribution="A", dominant_distribution_proportion=0.5)
                    else:
                        for j, filler_distribution in enumerate(filler_distributions):
                            if "variable_filler_distributions_no_subtract" in filler_type: 
                                new_filler_embedding[j,:] = embedding_util.create_word_vector(filler_distribution="C")
                            elif "variable_filler_distributions_one_distribution" in filler_type:
                                new_filler_embedding[j,:] = embedding_util.create_word_vector(filler_distribution=filler_distribution, dominant_distribution_proportion=1)
                            elif "variable_filler_distributions_all_randn_distribution" in filler_type:
                                new_filler_embedding[j,:] = embedding_util.create_word_vector(filler_distribution="randn")
                            elif "variable_filler_distributions_A" in filler_type:
                                new_filler_embedding[j,:] = embedding_util.create_word_vector(filler_distribution="A", dominant_distribution_proportion=1)
                            elif "variable_filler_distributions_B" in filler_type:
                                new_filler_embedding[j,:] = embedding_util.create_word_vector(filler_distribution="B", dominant_distribution_proportion=1)
                            elif "variable_filler_distributions_5050_AB" in filler_type:
                                new_filler_embedding[j,:] = embedding_util.create_word_vector(filler_distribution="B", dominant_distribution_proportion=0.5)
                            else:
                                new_filler_embedding[j,:] = embedding_util.create_word_vector(filler_distribution=filler_distribution)
                else:
                    for j in range(num_fillers):
                        new_filler_embedding[j,:] = embedding_util.create_word_vector()
                # Replace filler embedding with new random embedding.
                filler_ix_X = np.where(np.isin(batchX[examplenum], filler_indices))
                new_embedding_ix_X = [filler_indices.index(i) for i in batchX[examplenum,filler_ix_X][0]]
                embeddingX[examplenum,filler_ix_X] = new_filler_embedding[new_embedding_ix_X]
                if "noise" in filler_type:
                    if np.random.rand() < noise_proportion:
                        print('noise trial')
                        queried_filler_index = query_to_filler_indices[str(batchX[examplenum, -1])]
                        queried_filler_indices = np.where(batchX[examplenum] == queried_filler_index)
                        if zero_vector_noise:
                            print('zero vector noise')
                            embeddingX[examplenum, queried_filler_indices] = np.zeros(padding_vector.shape)
                        else:
                            embeddingX[examplenum, queried_filler_indices] = padding_vector
                new_embedding_ix_y = [filler_indices.index(batchy[examplenum])]
                embeddingy[examplenum] = new_filler_embedding[new_embedding_ix_y]
                # Append embedding to original embedding identifying response.
                epoch_embedding = np.concatenate((epoch_embedding, new_filler_embedding), axis=0)
            yield embeddingX, embeddingy, epoch_embedding

def shift_inputs(batchX, experiment_name, return_padding_location=False):
    # NOTE: Padding indices manually determined using word list saved by experiment creators.
    padding_index = experiment_parameters['padding_indices'][experiment_name]
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
