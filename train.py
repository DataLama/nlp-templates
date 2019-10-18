import os
import re
import argparse
import subprocess
import json
import logging
import pickle
import pandas as pd
import tensorflow as tf
import tensorflow.keras as ks
from pathlib import Path
from model.data import get_filenames, preprocess_fn
from model.net import keras_model_fn, keras_model_fn_cpu
from utils.utils import Config

## eager_execution and logging
tf.enable_eager_execution()
logging.getLogger().setLevel(logging.INFO)
tf.logging.set_verbosity(tf.logging.INFO)

## GPU memory utilization
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

## define input functions
def input_fn(batch_size, channel_name, train_ex_per_epoch, max_len):
    fn_dataset = get_filenames(channel_name)
    dataset = fn_dataset.interleave(lambda filepath: tf.data.TextLineDataset(filepath).skip(1), cycle_length=len(os.listdir(channel_name)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if 'train' in channel_name:
        buffer_size = int(train_ex_per_epoch * 0.4) + 3 * batch_size
        dataset = dataset.shuffle(buffer_size = buffer_size) # buffer size
    dataset = dataset.map(lambda x: preprocess_fn(x, max_len), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)

#### main function
def main(args):
    ## get configs
    data_dir = Path(args.data_dir)
    util_dir = Path(args.util_dir)
    data_config = Config(json_path = data_dir)
    model_config = Config(json_path = f'{args.model_dir}/params.json')
    utils_config = Config(json_path = util_dir)

    ## parse to data
    train_ex_per_epoch = sum([len(pd.read_csv('data/train/'+name)) for name in os.listdir('data/train')])
    
    ## make checkpoint_dir
    if os.path.exists(f'{args.model_dir}/checkpoints'):
        subprocess.run([f'rm -rf {args.model_dir}/checkpoints'], shell = True)
        subprocess.run([f'rm -rf {args.model_dir}/saved_model'], shell = True)
    os.mkdir(f'{args.model_dir}/checkpoints')

    ## load vocab and pretrained vectors
    with open(utils_config.vocab,'rb') as f:
        vocab = pickle.load(f)
    VOCAB_SIZE, EMBEDDING_SIZE= vocab.embedding.shape
    embeddings = [vocab.embedding]
     
    ## getting data
    logging.info("==== getting data ====")
    train_dataset = input_fn(model_config.batch_size, data_config.train, train_ex_per_epoch, model_config.max_len)
    validation_dataset = input_fn(model_config.batch_size, data_config.validation, train_ex_per_epoch, model_config.max_len)
    test_dataset = input_fn(model_config.batch_size, data_config.test, train_ex_per_epoch, model_config.max_len)
    
    ## configuring model
    logging.info("==== configuring model ====")
    model = keras_model_fn(model_config.model_name, model_config.learning_rate, VOCAB_SIZE, EMBEDDING_SIZE, embeddings)
    callbacks = [
        ks.callbacks.EarlyStopping(monitor = 'val_loss', min_delta=0.0001, patience = 20, restore_best_weights=True), # earlystopping
        ks.callbacks.ModelCheckpoint(args.model_dir + '/checkpoints/ckpt-loss-{epoch}.h5', monitor = 'val_loss', mode = 'min', save_best_only=True),
        ks.callbacks.ModelCheckpoint(args.model_dir + '/checkpoints/ckpt-top3-{epoch}.h5', monitor = 'val_categorical_accuracy', mode = 'max', save_best_only=True)]
    
    ## start training 
    logging.info("Starting training")
    hist = model.fit(train_dataset, validation_data=validation_dataset, epochs = model_config.epochs ,callbacks=callbacks) 
    score = model.evaluate(test_dataset) 
    logging.info(f'Test loss: {round(score[0], 4)} - Test categorical_accuracy: {round(score[1], 4)}')
    
    ## save best model
    # best-model-weights gpu
    model.save_weights(f'{args.model_dir}/best_model_weights.h5')
    print('best-weight-success')
    # best-model cpu
    cpu_model = keras_model_fn_cpu(f'{model_config.model_name}-cpu', model_config.learning_rate,VOCAB_SIZE, EMBEDDING_SIZE)
    cpu_model.load_weights(f'{args.model_dir}/best_model_weights.h5')
    cpu_model.save(f'{args.model_dir}/best_model_cpu.h5')

    # saved_model cpu
    tf.keras.experimental.export_saved_model(cpu_model, f'{args.model_dir}/saved_model/1/')
    
    ## save and send to s3
    experiment = dict()
    experiment['train'] = pd.DataFrame(hist.history).to_dict('r')
    experiment['test'] = pd.DataFrame({x:[score[i]] for i, x in enumerate(hist.params['metrics']) if 'val' not in x}).to_dict('r')
    with open(f'{args.model_dir}/history.json','w') as f:
        json.dump(experiment, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/config.json', help="Directory containing config.json of data")
    parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing config.json of model")
    parser.add_argument('--util_dir', default='utils/config.json', help="Directory containing config.json of utils_data")
    args = parser.parse_args()
    main(args)