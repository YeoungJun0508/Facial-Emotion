# 감정인식 vit모델

허깅페이스에 올라온 [감정인식 vit모델 ](https://huggingface.co/dima806/facial_emotions_image_detection)

이 vit모델은 [Facial Emotion Expressions](https://www.kaggle.com/datasets/samaneheslamifar/facial-emotion-expressions) 데이터 셋을 사용했음.

ViT 모델은 이미지를 패치로 분할하고, 각 패치를 토큰으로 처리하여 Transformer 아키텍처를 사용해 이미지 분류를 수행.

Self-Attention 메커니즘을 활용하여 이미지 특징을 추출하고, 최종적으로 선형 레이어를 통해 감정 클래스를 분류

(각 위치의 특성들 간의 관계를 계산)
## 모델 구성 (config)

모델 아키텍처: ViTForImageClassification

이미지 크기: 224x224

감정 클래스: 총 7개의 클래스 (sad, disgust, angry, neutral, fear, surprise, happy)

모델의 주요 하이퍼파라미터:

hidden_size: 768

num_attention_heads: 12

num_hidden_layers: 12

patch_size: 16

intermediate_size: 3072

입력 채널 수: 3 (RGB)

입력 이미지 크기

height: 224
width: 224


![image](https://github.com/YeoungJun0508/Facial-Emotion/assets/145903037/372cd64a-3073-4f6c-b7c3-943bb9e44eb0) ![image](https://github.com/YeoungJun0508/Facial-Emotion/assets/145903037/9ab5c9c2-044d-447f-905d-64f31be79a3b)

 ViT 모델은 이미지를 고정된 크기로 resize하고 normalization을 적용하여 모델에 입력. 

이 모델은 전체 이미지를 바탕으로 감정을 예측하며, 얼굴의 특정 부분을 분리하여 처리하지 않음.

ViTFeatureExtractor 함수로 입력 사이즈에 맞게 전처리해서 넣음.


### num_hidden_layers:
많은 레이어는 모델이 더 복잡한 패턴을 학습할 수 있게 하지만, 계산 비용과 학습 시간이 증가.

### patch_size: 16: 
입력 이미지가 작은 패치로 분할되어 각 패치가 Transformer 모델에 개별 입력으로 처리됨.

작은 패치는 더 많은 디테일을 포착할 수 있지만, 계산 비용이 증가.


### intermediate_size: 3072 
간 레이어는 입력 벡터를 더 높은 차원으로 매핑하여 비선형성을 추가하고, 모델의 표현력을 증가.


### 입력 데이터 전처리 방법 (processor)

do_resize: True - 이미지 크기를 조정

size: {'height': 224, 'width': 224} - 목표 이미지 크기

do_rescale: True - 이미지 값을 0~1 범위로 조정

rescale_factor: 0.00392156862745098 (1/255)

do_normalize: True - 이미지 평균값과 표준편차를 사용해 정규화

image_mean: [0.5, 0.5, 0.5] - 정규화할 때 사용할 평균값

image_std: [0.5, 0.5, 0.5] - 정규화할 때 사용할 표준편차





### 구성 요소

#### ViTEmbeddings
- **PatchEmbeddings**: 입력 이미지를 패치(patch)로 분할하고, 각 패치를 임베딩(embedding)하여 모델에 입력.
- **PositionalEmbeddings**: 패치의 위치 정보를 학습하기 위한 위치 임베딩을 추가.

#### ViTEncoder
- **ViTLayer**: 여러 개의 ViT 레이어로 구성.
  - **ViTSdpaAttention**: Multi-head Self-Attention 메커니즘을 사용하여 입력에 대한 어텐션을 계산.
  - **ViTIntermediate**: Feedforward 네트워크를 통해 어텐션 출력을 중간 특성으로 변환.
  - **ViTOutput**: 어텐션 출력을 다음 레이어에 전달하기 전에 최종 특성으로 변환.
  - **LayerNorm**: 레이어 정규화를 통해 입력 데이터를 정규화.

#### Classifier
- **Linear Layer**: 최종 ViTOutput에서 나온 특성을 감정 클래스로 분류하기 위한 선형 레이어.


이 모델은 이미지를 입력으로 받아 감정을 분류하는 ViT 기반의 이미지 분류 모델입니다.



