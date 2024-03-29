"""Helper modules for embeddings."""
import numpy as np
import tensorflow as tf

from hard_coded_things import embedding_size

def normalize(a):
    """Return normalized vector."""
    return a / np.linalg.norm(a)


def create_word_vector(filler_distribution=None, normalize_filler_distribution=True, dominant_distribution_proportion=0.9):
    """Returns a random Gaussian vector with embedding_size dimensions."""
    vector = np.random.normal(size=embedding_size)
    if filler_distribution == "A":
        if np.random.rand() < dominant_distribution_proportion:
            vector[::2] += 0.5
        else:
            vector[::2] -= 0.5
    elif filler_distribution == "B":
        if np.random.rand() < dominant_distribution_proportion:
            vector[::2] -= 0.5
        else:
            vector[::2] += 0.5
    elif filler_distribution == "C":
        if np.random.rand() < dominant_distribution_proportion:
            vector[::2] += 0.5
    elif filler_distribution == "randn":
        vector[::2] += 0
    return normalize(vector)


def get_01_accuracy(prediction, actual, embedding):
    """Returns percentage of correct predictions.

    Args:
        prediction: [batch_size x num_dimensions_per_word] tensor of predictions.
        actual: [batch_size x num_dimensions_per_word] tensor of correct outputs.
        embedding: [num_words x embedding_size] tensor of vector word representations.

    Returns:
        tensor containing percentage of correct predictions.
    """
    actual_indices = get_closest_cosinesimilarity(actual, embedding)
    prediction_indices = get_closest_cosinesimilarity(prediction, embedding)
    return tf.reduce_mean(tf.cast(tf.equal(actual_indices, prediction_indices), tf.float32))


def get_closest_cosinesimilarity(batch_array, embedding):
    """Returns index of closest words to vectors in batch_array.

    Args:
        batch_array: [batch_size x num_dimensions_per_word] tensor of vectors.
        embedding: [num_words x embedding_size] tensor of vector word representations.

    Returns:
        [batch_size] tensor containing indices of words in embedding that are
            closest to each vector in batch_array.
    """
    batch_array = tf.expand_dims(batch_array, axis=0) if len(batch_array.shape) < 2 else batch_array  # expand dims if batch size 1
    normed_embedding = tf.cast(tf.nn.l2_normalize(embedding, dim=1), tf.float32)
    normed_array = tf.cast(tf.nn.l2_normalize(batch_array, dim=1), tf.float32)
    cosine_similarity = tf.matmul(normed_array, tf.transpose(normed_embedding, [1, 0]))
    return tf.argmax(cosine_similarity, 1)


def get_corpus_index(vector, embedding):
    """Returns index of the word in a given embedding that is closest to a given vector."""
    return get_closest_cosinesimilarity(vector, embedding).eval()[0]
