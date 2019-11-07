import os
import subprocess
import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import layers

## model_fn -> 모델부터 다시 제대로 보자.
def keras_model_fn(model_config, vocab_size, embedding_size, embeddings):
    """GPU version of Stacked Bi-LSTM and Bi-GRU with Two Fasttext
    """
    ## hyperparams
    model_name = model_config['model_name']
    num_class = model_config['num_class']
    lstm_hs = model_config['lstm_hs']
    gru_hs = model_config['gru_hs']
    learning_rate = model_config['learning_rate']
    
    ## build model - , weights=[embeddings[1]]
    inputs = ks.Input(shape=(None,), dtype='int32', name='inputs')
    embedded_sequences_ft1 = layers.Embedding(vocab_size, embedding_size, trainable = True, mask_zero = False)(inputs)
    embedded_sequences_ft2 = layers.Embedding(vocab_size, embedding_size, trainable = True, mask_zero = False)(inputs)
    concat_embed = layers.concatenate([embedded_sequences_ft1 ,embedded_sequences_ft2])
    concat_embed = layers.SpatialDropout1D(0.5)(concat_embed)
    x = layers.Bidirectional(layers.CuDNNLSTM(lstm_hs, return_sequences = True))(concat_embed)
    x, x_h, x_c = layers.Bidirectional(layers.CuDNNGRU(gru_hs, return_sequences = True, return_state = True))(x)
    x_1 = layers.GlobalMaxPool1D()(x)
    x_2 = layers.GlobalAvgPool1D()(x)
    x_out = layers.concatenate([x_1 ,x_2, x_h])
    x_out = layers.BatchNormalization()(x_out)
    outputs = layers.Dense(num_class, activation = 'softmax', name = 'outputs')(x_out) # outputs
    model = ks.Model(inputs, outputs, name = model_name)
    
    ## compile
    model.compile(loss = 'categorical_crossentropy', 
                  optimizer = ks.optimizers.Adam(lr=learning_rate, clipnorm=.25, beta_1=0.7, beta_2=0.99), 
                  metrics = ['categorical_accuracy', ks.metrics.TopKCategoricalAccuracy(k=3)]) # metric what?
    return model


## model_fn
def keras_model_fn_cpu(model_config, vocab_size, embedding_size, embeddings):
    """ CPU version of Stacked Bi-LSTM and Bi-GRU with Two Fasttext
    """
    ## hyperparams
    model_name = model_config['model_name']
    num_class = model_config['num_class']
    lstm_hs = model_config['lstm_hs']
    gru_hs = model_config['gru_hs']
    learning_rate = model_config['learning_rate']
    
    with tf.device('/cpu:0'):
        ## build model
        inputs = ks.Input(shape=(None,), dtype='int32', name='inputs')
        embedded_sequences_ft1 = layers.Embedding(vocab_size, embedding_size, trainable = False, mask_zero = False)(inputs)
        embedded_sequences_ft2 = layers.Embedding(vocab_size, embedding_size, trainable = False, mask_zero = False)(inputs)
        concat_embed = layers.concatenate([embedded_sequences_ft1 ,embedded_sequences_ft2])
        concat_embed = layers.SpatialDropout1D(0.5)(concat_embed)
        x = layers.Bidirectional(layers.LSTM(lstm_hs,recurrent_activation = 'sigmoid', return_sequences = True))(concat_embed)
        x, x_h, x_c = layers.Bidirectional(layers.GRU(gru_hs, reset_after = True, recurrent_activation = 'sigmoid', return_sequences = True, return_state = True))(x)
        x_1 = layers.GlobalMaxPool1D()(x)
        x_2 = layers.GlobalAvgPool1D()(x)
        x_out = layers.concatenate([x_1 ,x_2, x_h])
        x_out = layers.BatchNormalization()(x_out)
        outputs = layers.Dense(num_class, activation = 'softmax', name = 'outputs')(x_out) # outputs
        model = ks.Model(inputs, outputs, name = model_name)

        ## compile
        model.compile(loss = 'categorical_crossentropy', 
                      optimizer=ks.optimizers.Adam(lr=learning_rate, clipnorm=.25, beta_1=0.7, beta_2=0.99), 
                      metrics=['categorical_accuracy', ks.metrics.TopKCategoricalAccuracy(k=3)])
        return model
    
## directory for checkpoint overwritable
def mkdir_checkpoint(model_dir):    
    if os.path.exists(f'{model_dir}/checkpoints'):
        subprocess.run([f'rm -rf {model_dir}/checkpoints'], shell = True)
        subprocess.run([f'rm -rf {model_dir}/saved_model'], shell = True)
    os.makedirs(f'{model_dir}/checkpoints', exist_ok = True) 