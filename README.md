# Basic NLP-templates with tf & keras.
Inspired from this awesome [repository](https://github.com/aisolab/nlp_implementation), build nlp-templates with tf & keras.

## Folder structure
```bash
.
├── data
│   ├── train
│   ├── validation
│   ├── test
│   ├── (pretrained)
│   ├── raw_train.csv
│   └── raw_test.csv
├── model
│   ├── data.py
│   ├── net.py
│   ├── (ops.py)
│   └── (metrics.py)
├── experiments
│   ├── {model_name}
│   │   ├── params.json
│   │   └── summary.json
│   └── ...
├── utils
│   ├── utils.py
│   ├── (synthesis_result.py)
│   └── ...
├── build_vocab.py
├── build_dataset.py
├── train.py
├── (evaluate.py)
├── (search_hyperparameter.py)
├── (config.json)
├── requirements.txt
└── README.md
```


## Main Components

## Quick Application
- data nsmc 사용함.
- model은 LSTM + GRU

# Future work
- tensorboard
    - Model Architecture
    - Learning Curve
    - some metrics
- Compare MeCab vs sentencepiece and apply.
- search_hyperparameters.py
- horovod


# References
- https://github.com/aisolab/nlp_implementation
- https://tykimos.github.io/warehouse/2019-7-4-ISS_2nd_Deep_Learning_Conference_All_Together_aisolab_file.pdf
- https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52644

