# Mission 2 - 굴뚝 높이 예측

## 의존성 설치
이 프로젝트는 `uv`를 사용하여 의존성을 관리합니다.

```bash
uv sync
```

또는 기존 pip를 사용하는 경우:
```bash
pip install -r requirements.txt
```

## 데이터셋 지정 방법
- 데이터셋은 `KompsatDataset` 클래스에 의해 자동으로 다운로드 되고 압축이 해제되어 코드에 필요한 형태로 배치됩니다.
- 데이터셋을 수동으로 지정하고자 하는 경우 `./data 폴더` 내부에 `TL_KS_BBOX.zip`, `TL_KS_LINE.zip`, `TS_KS.zip`, `VL_KS_BBOX.zip`, `VL_KS_LINE.zip`, `VS_KS.zip` 파일을 직접 넣어주시면 됩니다.
