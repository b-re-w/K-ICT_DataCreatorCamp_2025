# Mission 1 - YOLOv8 굴뚝 탐지

2025 데이터 크리에이터 캠프 Mission 1 프로젝트


**팀명**: 최후의 인공지능

## 프로젝트 개요

YOLOv8 모델을 사용하여 굴뚝(chimney)을 탐지하는 객체 탐지 프로젝트입니다.


## 설치 방법

### 1. uv를 이용한 설치 (권장)

```bash
# uv 설치 (설치되지 않은 경우)
pip install uv

# 의존성 설치
uv pip install -r requirements.txt
```

### 2. pip를 이용한 설치

```bash
pip install -r requirements.txt
```

## 데이터셋 준비

### 데이터셋 구조

프로젝트는 다음과 같은 데이터셋 구조를 사용합니다:

```
dataset/
├── train/
│   ├── images/          # 학습용 이미지 파일들
│   └── labels/          # VIA 포맷의 JSON 레이블 파일들
└── valid/
    ├── images/          # 검증용 이미지 파일들
    └── labels/          # VIA 포맷의 JSON 레이블 파일들
```

### 데이터셋 준비 단계

1. **이미지 및 레이블 파일 배치**
   - 학습 이미지를 `dataset/train/images/` 폴더에 배치
   - 학습 레이블(JSON)을 `dataset/train/labels/` 폴더에 배치
   - 검증 이미지를 `dataset/valid/images/` 폴더에 배치
   - 검증 레이블(JSON)을 `dataset/valid/labels/` 폴더에 배치

2. **레이블 포맷 변환**
   
   노트북의 섹션 2.1을 실행하여 VIA JSON 레이블을 YOLO 포맷으로 변환합니다:
   
   ```python
   convert_folder_via_jsons_to_yolo_txts(
       json_dir="./dataset/valid/labels",      # JSON 파일들이 있는 폴더
       images_dir="./dataset/valid/images",    # 이미지 폴더
       out_labels_dir="./dataset/valid/yolo_labels"  # YOLO 포맷 레이블 저장 폴더
   )
   ```
   
   이 함수는:
   - VIA 포맷의 JSON 파일을 읽어옵니다
   - 각 bounding box를 YOLO 포맷으로 변환합니다 (클래스 ID, 중심 x, 중심 y, 너비, 높이)
   - `yolo_labels` 폴더에 `.txt` 파일로 저장합니다

3. **Dataset YAML 파일 생성**
   
   노트북의 섹션 2.2를 실행하여 `dataset.yaml` 파일을 생성합니다:
   
   ```python
   dataset = {
       'path': '/root/jupyter/K-ICT_data_creator/dataset/',  # 데이터셋 루트 경로
       'train': 'train/images',  
       'val': 'test/images', 
       'test': 'test/images',
       'nc': 1,                  # 클래스 개수
       'names': ['chimney']      # 클래스 이름
   }
   ```
   
   **주의**: `path` 값을 실제 데이터셋이 위치한 절대 경로로 수정해야 합니다.

## 모델 학습

### 학습 실행

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

results = model.train(
    data='dataset.yaml', 
    seed=0,
    epochs=300, 
    imgsz=512,
    project='results',
    name='yolov8s_512',
    tracker="wandb",
    val=True
)
```

### 학습 파라미터

- **모델**: YOLOv8s (Small)
- **Epochs**: 300
- **이미지 크기**: 512x512
- **시드**: 0 (재현성을 위한 고정)
- **추적**: Weights & Biases (wandb)

## 모델 평가

학습된 모델을 평가하려면:

```python
from ultralytics import YOLO

model = YOLO("./results/yolov8s_512/weights/last.pt")

test_metrics = model.val(
    source="dataset.yaml",
    split='test',
    conf=0.25,    # confidence threshold
    iou=0.50,     # NMS IoU threshold
    max_det=100   # max detections
)
```
