분류
- data, experiments -> 가변적인 데이터
- model, utils -> 정적인 파일

사용법.
1. data폴더를 생성하고, data폴더에 config.json, train.csv, test.csv를 저장함.
    - config.json은 raw_train, raw_test에 대한 경로와 데이터를 split할 num_row를 정의함.


---
TODO LIST
- 임베딩 분석. 생성되는 vocab 파일과 실제 모델의 embedding 간의 차이 확인 및 이해.
- apply fasttext.
- adam -> radam optimizer.
- apply horovod to manipulate GPU.
- hyperparameter search.
    - SHAC 구현.
- model analysis.
    - metric 리스트업 및 코드 구현.
- serving env 구성 방법 고민.