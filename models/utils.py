import tensorflow as tf

# Create three `tf.data.Iterator` objects

def make_iterator(CVs, labels, batch_size, shuffle_and_repeat=False):
    """function that creates a `tf.data.Iterator` object"""
    dataset = tf.data.Dataset.from_tensor_slices((CVs, labels))
    if shuffle_and_repeat:
        dataset = dataset.apply(
            tf.data.experimental.shuffle_and_repeat(buffer_size=1000))
    
    def parse(CV, label):
        """function that returns cv and associated gender in a queriable format"""
        return {'CV': CV, 'label': label}

    dataset = dataset.apply(tf.data.experimental.map_and_batch(
        map_func=parse, batch_size=batch_size, num_parallel_batches=8))

    if shuffle_and_repeat:
        return dataset.make_one_shot_iterator()
    else:
        return dataset.make_initializable_iterator()