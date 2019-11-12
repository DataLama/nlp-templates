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
from model.data import get_filenames, preprocess_fn, input_fn
from model.net import keras_model_fn, keras_model_fn_cpu, mkdir_checkpoint



def main():
    #### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/config.json', help="Configure file which contains the directory information of Datasets.")
    parser.add_argument('--model_dir', default='experiments/base_model', help="Directory where the ouputs are saved.")
    args = parser.parse_args()
    
    #### Setting for TensorFlow 1.14 eager mode and GPU memory utilization.
    with open(f'{args.model_dir}/hyper_params.json') as f:
        model_config = json.load(f)
    
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{model_config['num_gpu']}"
    tf.enable_eager_execution()
    logging.getLogger().setLevel(logging.INFO)
    tf.logging.set_verbosity(tf.logging.INFO)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    
    print("==== Load Configs and Define Variables ====")
    mkdir_checkpoint(args.model_dir)
    with open(args.data_dir) as f:
        data_config = json.load(f)
    with open(data_config['vocab'],'rb') as f:
        vocab = pickle.load(f)  
    VOCAB_SIZE, EMBEDDING_SIZE = vocab.embedding[0].shape
    EMBEDDINGS = vocab.embedding
    tepe = sum([len(pd.read_csv('data/train/'+name, engine='python')) for name in os.listdir('data/train') if 'train' in name]) # the number of train examples per epoch.

    print("==== Build Data Pipelines ====")
    train_dataset = input_fn(model_config['batch_size'], data_config['train'], tepe, model_config['max_len'])
    validation_dataset = input_fn(model_config['batch_size'], data_config['validation'], tepe, model_config['max_len'])
    test_dataset = input_fn(model_config['batch_size'], data_config['test'], tepe, model_config['max_len'])
    
    print("==== Configure and Train the Model ====")
    callbacks = [
        ks.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.0001, patience = 10, restore_best_weights = True), # earlystopping
        ks.callbacks.ModelCheckpoint(args.model_dir + '/checkpoints/ckpt-loss-{epoch}.h5', monitor = 'val_loss', mode = 'min', save_best_only=True),
        ks.callbacks.ModelCheckpoint(args.model_dir + '/checkpoints/ckpt-top3-{epoch}.h5', monitor = 'val_categorical_accuracy', mode = 'max', save_best_only=True)]
    
    model = keras_model_fn(model_config ,VOCAB_SIZE, EMBEDDING_SIZE, EMBEDDINGS) #@@@@ change the function

    hist = model.fit(train_dataset, validation_data = validation_dataset, epochs = model_config['epochs'], callbacks = callbacks) 
    score = model.evaluate(test_dataset)
    print(f'Test loss: {round(score[0], 4)} - Test categorical_accuracy: {round(score[1], 4)} - Test top_k_categorical_accuracy: {round(score[2], 4)}')
    
    ## Save Model
    model.save_weights(f'{args.model_dir}/best_model_weights.h5')
    cpu_model = keras_model_fn_cpu(model_config, VOCAB_SIZE, EMBEDDING_SIZE, EMBEDDINGS)
    cpu_model.load_weights(f'{args.model_dir}/best_model_weights.h5')
    cpu_model.save(f'{args.model_dir}/best_model_cpu.h5')
    tf.keras.experimental.export_saved_model(cpu_model, f'{args.model_dir}/saved_model/1/')
    
    ## Save Train Metrics
    experiment = dict()
    experiment['train'] = pd.DataFrame(hist.history).to_dict('r')
    experiment['test'] = pd.DataFrame({x:[score[i]] for i, x in enumerate(hist.params['metrics']) if 'val' not in x}).to_dict('r')
    with open(f'{args.model_dir}/history.json','w') as f:
        json.dump(experiment, f)

if __name__ == '__main__':
    main()