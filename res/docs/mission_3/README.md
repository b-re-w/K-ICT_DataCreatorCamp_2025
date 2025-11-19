# Mission 3 - 세그멘테이션 프로젝트

## 프로젝트 설정

### 의존성 설치
이 프로젝트는 `uv`를 사용하여 의존성을 관리합니다.

```bash
# uv로 의존성 동기화
uv sync
```

또는 기존 pip를 사용하는 경우:
```bash
pip install -r requirements.txt
```

## 데이터셋 폴더 구조 안내
### 개요
`mission3.ipynb`에서 사용되는 세그멘테이션(Segmentation) 데이터셋을 위한 `./dataset` 폴더 배치 방법을 설명합니다.

## 폴더 구조
```
./dataset/
├── train_SN/
│   ├── images/
│   │   ├── SN10_CHN_00001_230409.tif
│   │   ├── SN10_CHN_00002_230409.tif
│   │   └── ... (TIF 위성 이미지 파일)
│   └── labels/
│       ├── SN10_CHN_00001_230409.tif
│       ├── SN10_CHN_00002_230409.tif
│       └── ... (TIF 라벨 마스크 파일)
└── valid_SN/
    ├── images/
    │   ├── SN10_CHN_05196_230921.tif
    │   └── ... (TIF 위성 이미지 파일)
    └── labels/
        ├── SN10_CHN_05196_230921.tif
        └── ... (TIF 라벨 마스크 파일)
```

## 배치 방법

### 1. 기본 폴더 생성
- `train_SN/`: 학습용 데이터
- `valid_SN/`: 검증용 데이터

### 2. 이미지/라벨 하위 폴더 생성
각 데이터셋 폴더 내에 `images/`와 `labels/` 폴더를 생성합니다.

### 3. 데이터 파일 배치
- **images 폴더**: 위성 이미지 TIF 파일 배치 (다중 밴드 RGB 이미지)
- **labels 폴더**: 대응되는 라벨 마스크 TIF 파일 배치 (세그멘테이션 마스크)
- 이미지와 라벨 파일명은 반드시 동일해야 함 (예: `SN10_CHN_00001_230409.tif`)
- `mission_3.ipynb`의 `load_tif_image()`, `load_tif_label()` 함수가 이 구조를 기반으로 데이터를 로드
- 라벨 값 매핑: 10 → 1 (foreground), 90 → 0 (background)으로 자동 변환됨
