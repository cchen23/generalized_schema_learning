"""Modules to generate train epochs."""
import numpy as np
import embedding_util

def generate_epoch(X, y, num_epochs, FLAGS, embedding):
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
        yield generate_batch(X, y, FLAGS, embedding)

def generate_batch(X, y, FLAGS, embedding):
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
            if FLAGS.function not in ["analyze", "probe_statistics"]:
                yield embedding[shift_inputs(X[start_index:end_index].squeeze(), FLAGS.experiment_name)], embedding[y[start_index:end_index].squeeze()], embedding
            else:
                yield embedding[X[start_index:end_index].squeeze()], embedding[y[start_index:end_index].squeeze()], embedding
        elif filler_type == "variable_filler":
            # NOTE: Filler indices manually determined using word list saved by experiment creators.
            if FLAGS.experiment_name == "variablefiller_gensymbolicstates_100000_1_testunseen_QSubject":
                FILLER_INDICES = [0, 5, 6, 13, 17, 18, 19, 20, 21, 22, 23, 24]
            elif FLAGS.experiment_name == "variablefiller_gensymbolicstates_100000_1_testunseen_AllQs":
                FILLER_INDICES = [0, 6, 7, 14, 19, 20, 24, 25, 26, 27, 28, 29]
            elif FLAGS.experiment_name == "storyv2_train20000_AllQs":
                FILLER_INDICES = [16, 12, 14, 17, 6, 8]
            else:
                raise ValueError("Unsupported experiment name")
            DIMS = 50
            batchX, batchy = X[start_index:end_index].squeeze(), y[start_index:end_index].squeeze()
            if FLAGS.function not in ["analyze", "probe_statistics"]: # Don't randomly shift inputs for decoding analysis.
                #print("flag function", FLAGS.function)
                batchX = shift_inputs(batchX, FLAGS.experiment_name)
            embeddingX, embeddingy = embedding[batchX], embedding[batchy]
            epoch_embedding = embedding
            for example_num in range(batch_size):
                # Create new random embedding for each filler.
                num_fillers = len(FILLER_INDICES)
                new_filler_embedding = np.empty((num_fillers, DIMS))
                for j in range(num_fillers):
                    new_filler_embedding[j,:] = embedding_util.create_word_vector()
                # Replace filler embedding with new random embedding.
                filler_ix_X = np.where(np.isin(batchX[example_num], FILLER_INDICES))
                new_embedding_ix_X = [FILLER_INDICES.index(i) for i in batchX[example_num,filler_ix_X][0]]
                embeddingX[example_num,filler_ix_X] = new_filler_embedding[new_embedding_ix_X]
                new_embedding_ix_y = [FILLER_INDICES.index(batchy[example_num])]
                embeddingy[example_num] = new_filler_embedding[new_embedding_ix_y]
                # Append embedding to original embedding identifying response.
                epoch_embedding = np.concatenate((epoch_embedding, new_filler_embedding), axis=0)
            yield embeddingX, embeddingy, epoch_embedding

def shift_inputs(batchX, experiment_name):
    # NOTE: Padding indices manually determined using word list saved by experiment creators.
    if experiment_name == "variablefiller_gensymbolicstates_100000_1_testunseen_QSubject":
        padding_index = 15
    elif experiment_name == "variablefiller_gensymbolicstates_100000_1_testunseen_AllQs":
        padding_index = 16
    elif experiment_name == "fixedfiller_gensymbolicstates_100000_1_AllQs":
        padding_index = 27
    elif experiment_name == "fixedfiller_gensymbolicstates_100000_1_testunseen_AllQs":
        padding_index = 27
    elif experiment_name == "storyv2_train20000_AllQs":
        padding_index = 18
    elif experiment_name == "generate_train3roles_testnewrole_10personspercategory_24000train_120test":
        padding_index = 58
    elif experiment_name == "generate_train3roles_testnewrole_100personspercategory_24000train_120test":
        padding_index = 418
    elif experiment_name == "generate_train3roles_testnewrole_1000personspercategory_24000train_120test":
        padding_index = 4018
    elif experiment_name == "generate_train3roles_testnewrole_withunseentestfillers_10personspercategory_24000train_120test":
        padding_index = 158
    elif experiment_name == "generate_train3roles_testnewrole_withunseentestfillers_100personspercategory_24000train_120test":
        padding_index = 518
    elif experiment_name in ["generate_train3roles_testnewrole_withunseentestfillers_1000personspercategory_24000train_120test", "generate_train3roles_testnewrole_withunseentestfillers_1000personspercategory_24000train_120test_add05fillers"]:
        padding_index = 4118
    elif "probestatisticsretention" in experiment_name:
        padding_index = 16017
    else:
        raise ArgumentError("Unsupported experiment name.")
    batch_size = batchX.shape[0]
    new_X = np.zeros(batchX.shape, dtype=np.int16)
    for i in range(len(batchX)):
        original_Xi = batchX[i]
        new_Xi = np.zeros(original_Xi.shape)
        first_padding_index = np.where(original_Xi == padding_index)[0][0]
        padding_location = np.random.choice(range(first_padding_index))
        new_Xi[:padding_location] = original_Xi[:padding_location]
        new_Xi[padding_location] = padding_index
        new_Xi[padding_location+1:first_padding_index+1] = original_Xi[padding_location:first_padding_index]
        new_Xi[first_padding_index+1:] = original_Xi[first_padding_index+1:]
        new_X[i] = new_Xi
    return new_X
