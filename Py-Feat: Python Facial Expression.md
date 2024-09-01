


# [FEAT API 문서](https://py-feat.org/pages/api.html)

## `detect_image` 메서드

**목표**: 하나 이상의 이미지 파일에서 FEX(Facial Expression Analysis) 정보를 감지합니다.

### 매개변수
- **input_file_list (list of str)**: 이미지 파일 경로 목록.
- **output_size (int)**: 모든 이미지를 비율을 유지하면서 리사이즈할 크기. 배치 크기가 1보다 크고 이미지 크기가 동일하지 않을 때 설정되지 않으면 오류가 발생합니다.
- **batch_size (int)**: 한 번에 처리할 이미지 배치의 수. 크기가 클수록 속도가 빠르지만 메모리 사용량이 많습니다. 배치에 포함된 모든 이미지의 크기는 동일해야 합니다.
- **num_workers (int)**: 데이터 로딩에 사용할 서브 프로세스의 수. 0이면 데이터는 메인 프로세스에서 로드됩니다.
- **pin_memory (bool)**: True인 경우, 데이터 로더는 텐서를 CUDA 핀 메모리로 복사하여 반환합니다. 데이터 요소가 사용자 정의 타입이거나 `collate_fn`이 사용자 정의 배치를 반환하는 경우 유용합니다.
- **frame_counter (int)**: 프레임 카운트의 시작 값.
- **face_detection_threshold (float)**: 얼굴 감지의 신뢰도에 따라 감지 결과를 보고하는 값 (0과 1 사이). 기본값은 0.5입니다.
- **face_identity_threshold (float)**: 얼굴 정체성 임베딩을 사용하여 사람의 유사성을 결정하는 값 (0과 1 사이). 기본값은 0.8입니다.
- **kwargs**: 각 감지기별로 특정 매개변수를 딕셔너리 형태로 전달할 수 있습니다. 예: `face_model_kwargs = {...}`, `au_model_kwargs = {...}`

### 반환값
- **Prediction results dataframe**: 감지 결과가 포함된 데이터프레임 (형식: `Fex`)

---

## `detect_landmarks` 메서드

**목표**: 이미지 또는 비디오 프레임에서 랜드마크를 감지합니다.

### 매개변수
- **frame (np.ndarray)**: 3D (단일 이미지) 또는 4D (여러 이미지) 배열 형식의 이미지.
- **detected_faces (array)**: 감지된 얼굴 좌표.

### 반환값
- **x와 y 랜드마크 좌표**: (1,68,2) 형식으로 반환됩니다.

### 반환 타입
- **list**

### 예시
```python
from feat import Detector
from feat.utils import read_pictures

# 이미지 파일 읽기
img_data = read_pictures(['my_image.jpg'])

# Detector 객체 생성
detector = Detector()

# 얼굴 감지
detected_faces = detector.detect_faces(img_data)

# 랜드마크 감지
detected_landmarks = detector.detect_landmarks(img_data, detected_faces)

```

# `feat.data` 모듈 문서

## `Fex` 클래스

`Fex` 클래스는 얼굴 표정(Fex) 데이터를 표현하는 데이터프레임을 확장한 클래스입니다.

### 생성자

```python
class Fex(DataFrame):
    def __init__(self, *args, **kwargs):
        ...
```
매개변수
filename (str, optional): 파일 경로.
detector (str, optional): Fex 추출에 사용된 소프트웨어 이름. 현재는 'Feat'만 지원됨.
sampling_freq (float, optional): 각 행의 샘플링 주파수 (Hz). 기본값은 None.
features (pd.DataFrame, optional): 각 Fex 행에 해당하는 특징.
sessions (array-like, optional): 특정 세션(예: 시험, 주제 등)과 관련된 고유 값. 1D 배열이어야 하며, 기본값은 None.
메서드
append(data, session_id=None, axis=0)

새로운 Fex 객체를 기존 객체에 추가합니다.
매개변수:
data: 추가할 Fex 인스턴스.
session_id: 세션 레이블.
axis: 추가할 축 (0: 행, 1: 열).
반환값: Fex 인스턴스.
baseline(baseline='median', normalize=None, ignore_sessions=False)

Fex 객체를 기준선에 참조합니다.
매개변수:
method: 'median', 'mean', 'begin', 또는 FexSeries 인스턴스.
normalize: 결과를 정규화할 방법 ('db', 'pct' 등).
ignore_sessions: 세션 정보를 무시할지 여부.
반환값: Fex 객체.
clean(detrend=True, standardize=True, confounds=None, low_pass=None, high_pass=None, ensure_finite=False, ignore_sessions=False, *args, **kwargs)

시계열 신호를 정리합니다.
매개변수:
confounds: 혼란 변수.
low_pass: 저주파 필터 컷오프 주파수 (Hz).
high_pass: 고주파 필터 컷오프 주파수 (Hz).
detrend: 시계열에서 추세를 제거할지 여부.
standardize: 신호를 단위 분산으로 조정할지 여부.
ensure_finite: 비유한 값(NANs 및 infs)을 0으로 대체할지 여부.
ignore_sessions: 세션 정보를 무시할지 여부.
반환값: 정리된 Fex 인스턴스.
compute_identities(threshold=0.8, inplace=False)

얼굴 임베딩을 사용하여 정체성을 계산합니다.
매개변수:
threshold: 임계값.
inplace: 기존 객체를 수정할지 여부.
반환값: 정체성 데이터.
decompose(algorithm='pca', axis=1, n_components=None, *args, **kwargs)

Fex 인스턴스를 분해합니다.
매개변수:
algorithm: 분해 알고리즘 ('pca', 'ica', 'nnmf', 'fa').
axis: 분해할 차원 (0 또는 1).
n_components: 구성 요소의 수. None이면 가능한 많은 구성 요소를 유지합니다.
반환값: 분해 파라미터의 딕셔너리.
distance(method='euclidean', **kwargs)

Fex 인스턴스 내의 행 간 거리 계산.
매개변수:
method: 거리 메트릭 타입 (scikit-learn 또는 scipy 메트릭).
반환값: 2D 거리 행렬.
downsample(target, **kwargs)

Fex 열을 다운샘플링합니다.
매개변수:
target: 다운샘플링 목표 (샘플 수).
kwargs: 추가 입력.
반환값: 다운샘플링된 Fex 인스턴스.
extract_boft(min_freq=0.06, max_freq=0.66, bank=8, *args, **kwargs)

Bag of Temporal features를 추출합니다.
매개변수:
min_freq: 최소 주파수.
max_freq: 최대 주파수.
bank: 템포랄 필터 뱅크의 수.
반환값: Morlet 웨이블렛 목록과 주파수 목록.
extract_max(ignore_sessions=False)

각 특징의 최대값을 추출합니다.
매개변수:
ignore_sessions: 세션 정보를 무시할지 여부.
반환값: 각 특징의 최대값.
extract_mean(ignore_sessions=False)

각 특징의 평균값을 추출합니다.
매개변수:
ignore_sessions: 세션 정보를 무시할지 여부.
반환값: 각 특징의 평균값.
extract_min(ignore_sessions=False)

각 특징의 최소값을 추출합니다.
매개변수:
ignore_sessions: 세션 정보를 무시할지 여부.
반환값: 각 특징의 최소값.
extract_sem(ignore_sessions=False)

각 특징의 표준 오차를 추출합니다.
매개변수:
ignore_sessions: 세션 정보를 무시할지 여부.
반환값: 각 특징의 표준 오차.
extract_std(ignore_sessions=False)

각 특징의 표준 편차를 추출합니다.
매개변수:
ignore_sessions: 세션 정보를 무시할지 여부.
반환값: 각 특징의 표준 편차.
extract_summary(mean=True, std=True, sem=True, max=True, min=True, ignore_sessions=False, *args, **kwargs)

여러 특징의 요약 정보를 추출합니다.
매개변수:
mean: 평균값 추출 여부.
std: 표준 편차 추출 여부.
sem: 표준 오차 추출 여부.
max: 최대값 추출 여부.
min: 최소값 추출 여부.
ignore_sessions: 세션 정보를 무시할지 여부.
반환값: 요약된 Fex 인스턴스.
extract_wavelet(freq, num_cyc=3, mode='complex', ignore_sessions=False)

복잡한 Morlet 웨이블렛으로 특징을 추출합니다.
매개변수:
freq: 추출할 주파수.
num_cyc: 웨이블렛의 사이클 수.
mode: 추출할 특징 ('complex', 'filtered', 'phase', 'magnitude', 'power').
ignore_sessions: 세션 정보를 무시할지 여부.
반환값: 웨이블렛 추출된 Fex 인스턴스.
plot_detections(faces='landmarks', faceboxes=True, muscles=False, poses=False, gazes=False, add_titles=True, au_barplot=True, emotion_barplot=True, plot_original_image=True)

Feat의 탐지 결과를 플로팅합니다.
매개변수:
faces: 'landmarks'로 얼굴 랜드마크를 그리거나 'aus'로 AU 시각화 모델을 사용합니다.
faceboxes: 감지된 얼굴 주위에 상자를 그릴지 여부.
muscles: AU 활동의 근육을 그릴지 여부.
poses: 얼굴 포즈를 그릴지 여부.
gazes: 시선 벡터를 그릴지 여부.
add_titles: 파일 이름을 제목으로 추가할지 여부.
au_barplot: AU 탐지의 서브플롯을 포함할지 여부.
emotion_barplot: 감정 탐지의 서브플롯을 포함할지 여부.
반환값: matplotlib 피규어의 리스트.
read_feat(filename=None, *args, **kwargs)

Feat 탐지 결과를 읽어옵니다.
매개변수:
filename: 읽어올 파일의 경로.
반환값: Fex 인스턴스.
to_numpy(as_series=False)

Fex 객체를 numpy 배열로 변환합니다.
매개변수:
as_series: Series로 반환할지 여부.
반환값: numpy 배열 또는 pandas Series.
transform(session=None, exclude=None, **kwargs)

데이터를 변형합니다.
매개변수:
session: 특정 세션의 데이터 변형.
exclude: 변형에서 제외할 열.
반환값: 변형된 Fex 인스턴스.
FexSeries 클래스
FexSeries 클래스는 Fex 객체의 시계열 데이터만을 포함하는 클래스입니다.

생성자
python
코드 복사
class FexSeries(pd.Series):
    def __init__(self, *args, **kwargs):
        ...
메서드
extract_dnn_features() -> pd.DataFrame

DNN 특징을 추출합니다.
반환값: DNN 특징이 포함된 pandas DataFrame.
extract_lstm_features() -> pd.DataFrame

LSTM 특징을 추출합니다.
반환값: LSTM 특징이 포함된 pandas DataFrame.
ImageDataset 클래스
ImageDataset 클래스는 이미지 데이터를 로드하고 관리하는 데 사용됩니다.

생성자
python
코드 복사
class ImageDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        ...
매개변수
root_dir (str): 이미지가 저장된 디렉토리의 경로.
transform (callable, optional): 이미지에 적용할 변환 함수.
메서드
__len__() -> int

데이터셋의 총 이미지 수를 반환합니다.
반환값: 데이터셋의 이미지 수.
__getitem__(idx: int) -> Tuple[Image, int]

주어진 인덱스에 해당하는 이미지와 레이블을 반환합니다.
매개변수:
idx: 이미지의 인덱스.
반환값: 이미지와 레이블의 튜플.
VideoDataset 클래스
VideoDataset 클래스는 비디오 데이터를 로드하고 관리하는 데 사용됩니다.

생성자
python
코드 복사
class VideoDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        ...
매개변수
root_dir (str): 비디오가 저장된 디렉토리의 경로.
transform (callable, optional): 비디오에 적용할 변환 함수.
메서드
__len__() -> int

데이터셋의 총 비디오 수를 반환합니다.
반환값: 데이터셋의 비디오 수.
__getitem__(idx: int) -> Tuple[Video, int]

주어진 인덱스에 해당하는 비디오와 레이블을 반환합니다.
매개변수:
idx: 비디오의 인덱스.
반환값: 비디오와 레이블의 튜플.
