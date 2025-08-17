# K-ICT_DataCreatorCamp_2025

### Dataset Download
Download from [GDrive](https://github.com/b-re-w/K-ICT_DataCreatorCamp_2025)
- Data: `TS_KS.zip & VS_KS.zip` (Mission 1/2), `TS_SN10_SN10.zip & VS_SN10_SN10.zip` (Mission 3)
- Label: `TL_KS_BBOX.zip & VL_KS_BBOX.zip` (Mission 1), `TL_KS_LINE.zip & VL_KS_LINE.zip` (Mission 2), `TL_SN10.zip & VL_SN10.zip` (Mission 3)


### Dataset Preview
#### Mission 1 (Object Detection)
```json
"shape_attributes": {'name': "rect", 'x': 336, 'y': 280, 'width': 45, 'height': 100}
```
- Metric: mAP@IoU = 0.5

#### Mission 2 (Height Estimation from Images)
```json
"region_attributes": {'chi_id': "1", 'chi_height_m': "187.28"}
```
- Unit: meter
- Metric: RMSE

#### Mission 3 (Semantic Segmentation)
```json
"shape_attributes": {'name': "rect", 'x': 336, 'y': 280, 'width': 45, 'height': 100}
```
- Metric: mIoU


### Experiments
#### Mission 1 (Detectron2 ver)
- 작은 객체이므로 Transformers DETR 사용하면 성능 안나올 것
- Detectorn2의 SwinT + Faster RCNN 프리트레인 불러와서 파인튜닝하는 방식으로

#### Mission 1 (Transformers ver)
- 근데 문제는 프리트레인 모델을 가져다 쓴다 생각하면 최신 SOTA 성능을 찍어둔걸 쓰는게 더 좋을수도
- RT-DETR 모델로 학습할 때는 640x640으로 하고, 추론할 때는 896x896 정도로 높여서 추론하면
- 작은 물체 감지 성능이 올라갈 수도? (구현이 간단하니 이거 먼저 해봐야지)

#### Mission 2 (Transformers ver)
- Transformers DINOv3 + 커스텀 리니어레이어

#### Mission 3 (Transformers ver)
- Segmentation에는 SwinT + UperNet 기반 모델이 성능이 높을 것으로 보임
- UperNetForSemanticSegmentation 프리트레인 파인튜닝해 사용하거나
- SegFormer-B2도 비교 실험 진행
