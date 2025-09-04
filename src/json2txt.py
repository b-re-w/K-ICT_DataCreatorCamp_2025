import os
import cv2
import glob
import json
import math
import shutil

# 데이터 디렉토리 설정
baseDataDir = './data/'  # 기본 데이터 디렉토리
saveDir = './data/chimney_yolo/'  # YOLO 라벨 파일 저장 디렉토리

# 처리 대상 폴더들 설정
dataConfigs = [
    {"dataDir": "test_data", "outYoloDir": "test", "outFcDir": "test_fc"},
    {"dataDir": "train_data", "outYoloDir": "train", "outFcDir": "test_fc_train"},
    {"dataDir": "val_data", "outYoloDir": "val", "outFcDir": "test_fc_val"},
]

chi_h_quant_step = 30
dataDistribution = {'0': 0, '30': 0, '60': 0, '90': 0, '120': 0, '150': 0, '180': 0, '210': 0, '240': 0, '270': 0, '300': 0}


def get_chimney_gt(imageList, labelLineDir, outYoloDir, outFcDir):
    global dataDistribution

    for imagePath in imageList:
        imgFilename = os.path.basename(imagePath)
        labelFilename = imgFilename.replace('jpg', 'json')
        labelLinePath = os.path.join(labelLineDir, labelFilename)
        outLabelYoloPath = os.path.join(outYoloDir, labelFilename.replace('json', 'txt'))
        outLabelFcPath = os.path.join(outFcDir, labelFilename.replace('json', 'txt'))
        outImgPath = os.path.join(outYoloDir, imgFilename)

        if not os.path.exists(labelLinePath):
            print(f"[WARN] Missing JSON files for {imgFilename}")
            continue

        rect_list = []
        keypoint_list = []

        img = cv2.imread(imagePath)
        if img is None:
            print(f"[ERR] Failed to load image: {imagePath}")
            continue

        with open(labelLinePath, encoding='UTF8') as f_line:
            labelLineData = json.load(f_line)

        keyValue = list(labelLineData.keys())[0]
        labelLineData = labelLineData[keyValue]

        # BBox 데이터 읽기
        for region in labelLineData['regions']:
            if region['shape_attributes']['name'] == 'polyline':
                bottom_x = region['shape_attributes']['all_points_x'][0]
                bottom_y = region['shape_attributes']['all_points_y'][0]
                top_x = region['shape_attributes']['all_points_x'][1]
                top_y = region['shape_attributes']['all_points_y'][1]
                chi_height_m = region['region_attributes']['chi_height_m']
                keypoint_list.append({'bottom_x': bottom_x, 'bottom_y': bottom_y,
                                      'top_x': top_x, 'top_y': top_y,
                                      'chi_height_m': chi_height_m})

                bbox_left = max(min(bottom_x, top_x) - 1, 0)
                bbox_top = max(min(bottom_y, top_y) - 1, 0)
                bbox_width = abs(bottom_x - top_x) + 2
                bbox_height = abs(bottom_y - top_y) + 2
                bbox_right = bbox_left + bbox_width
                bbox_bottom = bbox_top + bbox_height
                rect_list.append({'bbox_width': bbox_width, 'bbox_height': bbox_height,
                                  'bbox_left': bbox_left, 'bbox_top': bbox_top,
                                  'bbox_right': bbox_right, 'bbox_bottom': bbox_bottom})

        imgHeight, imgWidth, _ = img.shape

        # 메타데이터 확인
        img_resolution = labelLineData['file_attributes'].get('img_resolution')
        img_roll_tilt = labelLineData['file_attributes'].get('img_roll_tilt')
        img_pitch_tilt = labelLineData['file_attributes'].get('img_pitch_tilt')

        if not (img_resolution and img_roll_tilt and img_pitch_tilt):
            print(f"[ERR] Missing metadata in {imagePath}")
            continue

        # 출력 디렉토리 생성
        os.makedirs(os.path.dirname(outLabelYoloPath), exist_ok=True)
        os.makedirs(os.path.dirname(outLabelFcPath), exist_ok=True)

        annotation_yolo = []
        annotation_fc = []

        if len(rect_list) == 0:  # 굴뚝 객체가 없을 경우
            # test_fc 폴더에는 기본값을 저장
            annotation_fc.append([0, 0, 0, 0, 0, img_resolution, img_roll_tilt, img_pitch_tilt])
            # test 폴더에는 9개의 0 값 저장 (YOLO가 인식하도록)
            annotation_yolo.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            for idx in range(len(rect_list)):
                wc = (rect_list[idx]['bbox_left'] + rect_list[idx]['bbox_right']) / 2 / imgWidth
                hc = (rect_list[idx]['bbox_top'] + rect_list[idx]['bbox_bottom']) / 2 / imgHeight
                w = rect_list[idx]['bbox_width'] / imgWidth
                h = rect_list[idx]['bbox_height'] / imgHeight
                kpt_x1 = keypoint_list[idx]['bottom_x'] / imgWidth
                kpt_y1 = keypoint_list[idx]['bottom_y'] / imgHeight
                kpt_x2 = keypoint_list[idx]['top_x'] / imgWidth
                kpt_y2 = keypoint_list[idx]['top_y'] / imgHeight

                annotation_yolo.append([0, wc, hc, w, h, kpt_x1, kpt_y1, kpt_x2, kpt_y2])

                annotation_fc.append([
                    keypoint_list[idx]['chi_height_m'],  # 굴뚝 높이
                    kpt_x1 * imgWidth, kpt_y1 * imgHeight,  # 하단 좌표
                    kpt_x2 * imgWidth, kpt_y2 * imgHeight,  # 상단 좌표
                    img_resolution, img_roll_tilt, img_pitch_tilt  # 메타데이터
                ])

        # YOLO TXT 저장
        with open(outLabelYoloPath, "w") as fp:
            for line in annotation_yolo:
                fp.write(" ".join(map(str, line)) + "\n")

        # Feature Center TXT 저장
        with open(outLabelFcPath, "w") as fp:
            for line in annotation_fc:
                fp.write(" ".join(map(str, line)) + "\n")

        # 이미지 복사 저장
        shutil.copyfile(imagePath, outImgPath)

    print(f"[INFO] Processed {len(imageList)} images")


if __name__ == "__main__":
    for config in dataConfigs:
        dataDir = os.path.join(baseDataDir, config["dataDir"])
        imageDir = os.path.join(dataDir, "KS")
        labelLineDir = os.path.join(dataDir, "KS_LINE")
        outYoloDir = os.path.join(saveDir, config["outYoloDir"])
        outFcDir = os.path.join(saveDir, config["outFcDir"])

        imageList = glob.glob(os.path.join(imageDir, '*.jpg'))
        os.makedirs(outYoloDir, exist_ok=True)
        os.makedirs(outFcDir, exist_ok=True)

        get_chimney_gt(imageList, labelLineDir, outYoloDir, outFcDir)

    print("Data conversion completed successfully!")
