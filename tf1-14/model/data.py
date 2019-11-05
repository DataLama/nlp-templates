import tensorflow as tf

## read filenames 
def get_filenames(channel_name):
    """
    channel_name : data/train
    """
    if 'train' in channel_name:
        return tf.data.Dataset.list_files(f'{channel_name}/*.csv', shuffle=True, seed=777)
    elif ('validation' in channel_name) or ('test' in channel_name):
            return tf.data.Dataset.list_files(f'{channel_name}/*.csv', shuffle=False)
    else:
        raise ValueError('Invalid data subset "%s"' % channel_name)
        
## tensor preprocess function
def preprocess_fn(line, max_len):
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
    # padding - 200 pad
    def true_fn():return tf.slice(tf.reshape(doc, shape=[-1]),[0],[max_len]) # 처음부터 200까지 자르기.
    def false_fn():
        pad = tf.reshape(tf.zeros(tf.subtract(tf.constant(max_len),tf.size(doc)), dtype=tf.int32), (1,-1))
        return tf.reshape(tf.concat([doc,pad],axis=1), shape=[-1]) # pad 붙이기
    doc = tf.cond(tf.size(doc) > max_len, true_fn, false_fn) #조건문..
    
    label = tf.one_hot(label, depth=63) # depth is number of class
    return doc, label