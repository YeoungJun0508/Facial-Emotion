


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
