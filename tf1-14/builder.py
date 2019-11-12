import os
import sys
import json
import pickle
import shutil
import itertools
import numpy as np
import pandas as pd
import gluonnlp as nlp
from pathlib import Path
from collections import Counter
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from utils.utils import split_fn, Vocab, Tokenizer

SP = nlp.data.SentencepieceTokenizer('utils/skt_tokenizer_78b3253a26.model')

class BuilderPtrEmbed(object):
    """Builder class"""
    
    def __init__(self, sp, config_path):
        """
        
        sp : gluonnlp.data.SentencepieceTokenizer object
        config : config info
        """
        self.sp = sp
        self.config_path = config_path
        self.cwd = Path.cwd()
        self.vocab = Vocab(self.sp.tokens, padding_token = '<pad>', unknown_token = '<unk>', bos_token = None, eos_token = None, token_to_idx = {'<unk>': 1})
        self.index = dict()

        with open(self.config_path) as f:
            self.config = json.load(f)
            
    def build_vocab(self):
        """
        """
        embedding_list = []
        
        for source_name in ['wiki.ko', 'cc.ko.300']:
            tmp_vocab = nlp.Vocab(counter = Counter(self.sp.tokens),  unknown_token = '<unk>', padding_token = '<pad>', min_freq = 1, bos_token = None, eos_token = None, token_to_idx = {'<unk>':1})
            embedding = nlp.embedding.create('fasttext', source = source_name)
            tmp_vocab.set_embedding(embedding)
            array = tmp_vocab.embedding.idx_to_vec.asnumpy()
            array[1] = array.mean(axis=0)
            embedding_list.append(array)
            OOV = int(((array == 0.).sum(axis=1) == array.shape[1]).sum())
            print(f"The number of OOV is {OOV} by {array.shape[0]}")
            self.index.update({"OOV":OOV})
        
        self.vocab.embedding = embedding_list
        
        self.index.update({'token2idx': self.vocab.token_to_idx})
        self.index.update({'idx2token': {v:k for k, v in self.vocab.token_to_idx.items()}})
    
    def save_vocab(self):
        vocab_file = 'data/vocab.pkl'
        
        with open(self.cwd / vocab_file, mode = 'wb') as io:
            pickle.dump(self.vocab, io)
            
        self.config.update({'vocab' : vocab_file})
        
    def _load_dataset(self, filepath):
        dataset = pd.read_csv(filepath, sep='\t')
        dataset.drop(['class_1','class_2'],axis=1, inplace=True) # preprocessing dataset
        dataset = dataset.loc[dataset['text'].isna().apply(lambda elm: not elm), :]
        return dataset
    
    def _mkdir(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok = True) 
        
    def build_dataset(self):
        #### load dataset
        dataset = self._load_dataset(self.cwd / self.config['raw_train'])
        
        label = sorted(dataset['class'].unique())
        self.index.update({'label2idx' : {c : i for i, c in enumerate(label)}})
        self.index.update({'idx2label' : {i : c for i, c in enumerate(label)}})
        dataset['class'] = [self.index['label2idx'][x] for x in dataset['class'].to_list()]
        
        train, validation = train_test_split(dataset, test_size=0.1, random_state=777)
        
        test = self._load_dataset(self.cwd / self.config['raw_test'])
        test['class'] = [self.index['label2idx'][x] for x in test['class'].to_list()]
        
        #### split and transform
        tokenizer = Tokenizer(vocab = self.vocab, split_fn = split_fn)
        self.holdout = [('train', train), ('validation', validation), ('test', test)]
        
        for _, ds in self.holdout:
            with Pool(processes = os.cpu_count()) as pool:
                ds['text'] = pool.map(tokenizer.split_and_transform, ds['text'].tolist())
    
    def save_dataset(self):
        for fn in ['train', 'validation', 'test']:
            self.config.update({fn : f'data/{fn}'})
            self._mkdir(self.config[fn])
        
        for name, ds in self.holdout:
            b = 0 
            for i in range((len(ds)//self.config['num_row']) + 1):
                e = (i+1) * self.config['num_row']
                ds.iloc[b:e].to_csv(self.cwd / f'data/{name}/{name}_{i}.csv', index = False)
                b = e
                
    def save_config(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f)
        with open('data/index.json', 'w') as f:
            json.dump(self.index, f)
            
def main():
    config_path = 'data/config.json'
    builder = BuilderPtrEmbed(SP, config_path)
    
    print('Now building vocab...')
    builder.build_vocab()
    builder.save_vocab()
    
    print('Now building dataset...')
    builder.build_dataset()
    builder.save_dataset()
    builder.save_config()
    print('SUCCESS')


if __name__ == '__main__':
    main()