import os, sys
import tarfile
import itertools
import pickle
import gluonnlp as nlp
import mxnet as mx
import numpy as np
import pandas as pd
import subprocess
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from konlpy.tag import Mecab
from collections import Counter
from multiprocessing import Pool
from utils.utils import Config
from utils.nlp_utils import Vocab, Tokenizer

# base
def split_fn(doc): return Mecab().morphs(doc) # mecab.morph
data_config = Config('data/config.json') # config for data.
cwd = Path.cwd() # current paths
min_freq = data_config.min_freq
num_row = data_config.num_row
utils_config = Config('utils/config.json') # empty

## document, label
print('==== step 1) Load dataset ====')
# Load train and build label indexer
filepath = cwd / data_config.raw_train
dataset = pd.read_csv(filepath, sep='\t')
label = sorted(dataset['label'].unique().tolist())
utils_config.label2idx = {c:i for i, c in enumerate(label)}
utils_config.idx2label = {i:c for i, c in enumerate(label)}

# train test split, label transform str -> int
dataset = dataset.loc[dataset['document'].isna().apply(lambda elm: not elm), :]
train, validation = train_test_split(dataset, test_size=0.1, random_state=777)
train['label'] = train['label'].apply(lambda x:utils_config.label2idx[x])
validation['label'] = validation['label'].apply(lambda x:utils_config.label2idx[x])

# Load test
filepath = cwd / data_config.raw_test
if os.path.exists(filepath):
    test = pd.read_csv(filepath, sep='\t')
    test = test.loc[test['document'].isna().apply(lambda elm: not elm), :]
    test['label'] = test['label'].apply(lambda x:utils_config.label2idx[x])
print('==== SUCCESS!!! ====')

print('==== step 2) build vocab ====')
# generate Vocab
with Pool(processes = os.cpu_count()) as pool:
    list_of_tokens = pool.map(split_fn, train['document'].tolist())
token_counter = Counter(itertools.chain.from_iterable(list_of_tokens))
list_of_tokens = [token_count[0] for token_count in token_counter.items() if token_count[1] >= min_freq]
list_of_tokens = sorted(list_of_tokens)

tmp_vocab = nlp.Vocab(counter=Counter(list_of_tokens),  unknown_token='<UNK>', padding_token='<PAD>',min_freq=1, bos_token=None, eos_token=None)
wiki = nlp.embedding.create('fasttext', source='wiki.ko')
tmp_vocab.set_embedding(wiki)
array = tmp_vocab.embedding.idx_to_vec.asnumpy()
vocab = Vocab(list_of_tokens, padding_token='<PAD>', unknown_token='<UNK>', bos_token=None, eos_token=None,unknown_token_idx=1)
vocab.embedding = array
vocab.embedding[1] = array.mean(axis=0) # unk를 전체 벡터의 평균으로 init
print(f'Vocab Size - {len(list_of_tokens)}')

# save Vocab
with open('utils/vocab.pkl', mode='wb') as io:
    pickle.dump(vocab, io)

utils_config.vocab = 'utils/vocab.pkl'
utils_config.token2idx = vocab.token_to_idx
utils_config.idx2token = vocab.idx_to_token
utils_config.save('utils/config.json')
print('==== SUCCESS!!! ====')

print('==== step 3) split and transform ====')
tokenizer = Tokenizer(vocab = vocab, split_fn = split_fn) # ft_koi랑 ft_cc_ko 어짜피 토큰 같음.

if os.path.exists(filepath):
    holdout = [('train', train), ('validation', validation), ('test', test)]
else:
    holdout = [('train', train), ('validation', validation)]

for _, ds in holdout:
    with Pool(processes = os.cpu_count()) as pool:
        ds['document'] = pool.map(tokenizer.split_and_transform, ds['document'].tolist())
print('==== SUCCESS!!! ====')

print('==== step 4) save ====')
# num_row 만큼 쪼개서 저장. 경로는 미리 만들어줘.
for name, ds in holdout:
    b = 0 
    if name == 'train':
        for i in range((len(ds)//num_row) + 1):
            e = (i+1) * num_row
            ds.iloc[b:e].to_csv(cwd / f'data/{name}/{name}_{i}.csv', index = False)
            b = e
    else:
        ds.to_csv(cwd / f'data/{name}/{name}.csv', index = False)
print('==== SUCCESS!!! ====')