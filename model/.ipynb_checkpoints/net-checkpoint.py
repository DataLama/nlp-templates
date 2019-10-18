import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import layers

## model_fn
def keras_model_fn(model_name,learning_rate, vocab_size, embedding_size, embeddings):
    """keras_model_fn receives hyperparameters from the training job and returns a compiled keras model.
    The model will be transformed into a TensorFlow Estimator before training and it will be saved in a 
    TensorFlow Serving SavedModel at the end of training.
    Args:
        hyperparameters: The hyperparameters passed to the SageMaker TrainingJob that runs your TensorFlow 
                         training script.
    Returns: A compiled Keras model
    """
    ## build model
    inputs = ks.Input(shape=(None,), dtype='int32', name='inputs')
    embedded_sequences = layers.Embedding(vocab_size, embedding_size, trainable = False, weights=[embeddings[0]], mask_zero=False)(inputs)
    concat_embed = layers.SpatialDropout1D(0.5)(embedded_sequences)
    x = layers.Bidirectional(layers.CuDNNLSTM(128, return_sequences=True))(concat_embed)
    x, x_h, x_c = layers.Bidirectional(layers.CuDNNGRU(64, return_sequences=True, return_state = True))(x)
    x_1 = layers.GlobalMaxPool1D()(x)
    x_2 = layers.GlobalAvgPool1D()(x)
    x_out = layers.concatenate([x_1 ,x_2, x_h])
    x_out = layers.BatchNormalization()(x_out)
    outputs = layers.Dense(63, activation = 'softmax', name = 'outputs')(x_out) # outputs
    model = ks.Model(inputs, outputs, name = model_name)
    
    ## compile
    # optimizer how?
    model.compile(loss = 'categorical_crossentropy', 
                  optimizer=ks.optimizers.Adam(lr=learning_rate, clipnorm=.25, beta_1=0.7, beta_2=0.99), 
                  metrics=['categorical_accuracy']) # metric what?
    return model


## model_fn
def keras_model_fn_cpu(model_name,learning_rate, vocab_size, embedding_size):
    """keras_model_fn receives hyperparameters from the training job and returns a compiled keras model.
    The model will be transformed into a TensorFlow Estimator before training and it will be saved in a 
    TensorFlow Serving SavedModel at the end of training.
    Args:
        hyperparameters: The hyperparameters passed to the SageMaker TrainingJob that runs your TensorFlow 
                         training script.
    Returns: A compiled Keras model
    """
    with tf.device('/cpu:0'):
        ## build model
        inputs = ks.Input(shape=(None,), dtype='int32', name='inputs')
        embedded_sequences = layers.Embedding(vocab_size, embedding_size, trainable = False, mask_zero=False)(inputs)
        concat_embed = layers.SpatialDropout1D(0.5)(embedded_sequences)
        x = layers.Bidirectional(layers.LSTM(128,recurrent_activation='sigmoid', return_sequences=True))(concat_embed)
        x, x_h, x_c = layers.Bidirectional(layers.GRU(64, reset_after=True, recurrent_activation='sigmoid',return_sequences=True, return_state = True))(x)
        x_1 = layers.GlobalMaxPool1D()(x)
        x_2 = layers.GlobalAvgPool1D()(x)
        x_out = layers.concatenate([x_1 ,x_2, x_h])
        x_out = layers.BatchNormalization()(x_out)
        outputs = layers.Dense(63, activation = 'softmax', name = 'outputs')(x_out) # outputs
        model = ks.Model(inputs, outputs, name = model_name)

        ## compile
        # optimizer how?
        model.compile(loss = 'categorical_crossentropy', 
                      optimizer=ks.optimizers.Adam(lr=learning_rate, clipnorm=.25, beta_1=0.7, beta_2=0.99), 
                      metrics=['categorical_accuracy']) # metric what?
        return model