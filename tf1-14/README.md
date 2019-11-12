# nlp-templates
- `tensorflow 1.14.0` 버전용 NLP templates. 
- `tf.enable_eager_execution` + `tf.data pipeline` + `tf.keras fit`을 바탕으로 쉽고, 빠르게 모델링을 할 수 있도록 구성함.
- Model Farm을 통해 다양한 **아키텍쳐**와 **하이퍼파라미터**의 모델을 쉽게 비교 가능함.
    - Model Farm에서 선택된 모델이 최종 모델이 됨.


### TO DO
- horovod 기반의 multi gpu 학습 구현.
- SHAC 기반의 hyperparameter search 알고리즘 구현.
- notebook 기반의 다양한 model, error analysis 기능 구현.

---
# 템플릿 구조
## 전체
- data, experiments -> 가변적인 데이터
- model, utils -> 정적인 파일

## config파일 관리
- config.json은 raw_train, raw_test에 대한 경로와 데이터를 split할 num_row를 정의함.

## outputs 설명.

## Model Farm

```bash
.
├── bi-LSTM+attention # model architectures
│   ├── data
│   ├── experiments # hyperparams
│   │   ├── ft300+bi-LSTM+attention
│   │   └── ...
│   ├── ...
│   ├── train.py
│   └── ...
├── CNN+bi-LSTM
│   ├── data
│   ├── experiments
│   │   ├── w2v300+CNN+bi-LSTM
│   │   └── ...
│   ├── ...
│   ├── train.py
│   └── ...
├── model_farm # Model Farm contains all model architecture and hyperparams.
│   ├── ft300+bi-LSTM+attention
│   ├── w2v300+CNN+bi-LSTM
│   └── ...
└── ...
```
