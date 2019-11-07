import os
import tensorflow as tf

## read filenames 
def get_filenames(channel_name):
    """
    channel_name : data/train
    """
    if 'train' in channel_name:
        return tf.data.Dataset.list_files(f'{channel_name}/*.csv', shuffle=True) #seed=777
    elif ('validation' in channel_name) or ('test' in channel_name):
            return tf.data.Dataset.list_files(f'{channel_name}/*.csv', shuffle=False)
    else:
        raise ValueError('Invalid data subset "%s"' % channel_name)
        
## tensor preprocess function
def preprocess_fn(line):
    """
    doc : string -> RaggedTensor
    label.
    """
    defs = [tf.constant([], dtype=tf.string) ,tf.constant([], dtype=tf.string), tf.constant([], dtype=tf.int32)] # input schema 정의.
    seq_no, doc, label = tf.io.decode_csv(line, record_defaults=defs)
    doc = tf.strings.regex_replace(doc, '\[','')
    doc = tf.strings.regex_replace(doc, '\]','')
    doc = tf.strings.split(doc, sep=',', result_type='RaggedTensor') # doc ~ RaggedTensor / [doc] ~ SparseTensor
    doc = tf.string_to_number([doc], tf.int32)
#     doc = tf.reshape(doc, shape=[-1])
    # padding - 200 pad
    def true_fn():return tf.slice(tf.reshape(doc, shape=[-1]),[0],[200]) # 처음부터 200까지 자르기.
    def false_fn():
        pad = tf.reshape(tf.zeros(tf.subtract(tf.constant(200),tf.size(doc)), dtype=tf.int32), (1,-1))
        return tf.reshape(tf.concat([doc,pad],axis=1), shape=[-1]) # pad 붙이기
    doc = tf.cond(tf.size(doc) > 200, true_fn, false_fn) #조건문..
    
    label = tf.one_hot(label, depth=51) # depth is number of class
    return doc, label

## tf.data.dataset pipeline
def input_fn(batch_size, channel_name, num_train_examples_per_epoch, max_len):
    fn_dataset = get_filenames(channel_name)
    dataset = fn_dataset.interleave(lambda filepath: tf.data.TextLineDataset(filepath).skip(1), cycle_length=len(os.listdir(channel_name)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if 'train' in channel_name:
        buffer_size = int(num_train_examples_per_epoch * 0.4) + 3 * batch_size
        dataset = dataset.shuffle(buffer_size = buffer_size) # buffer size
    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.padded_batch(batch_size, dataset.output_shapes)
    return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)