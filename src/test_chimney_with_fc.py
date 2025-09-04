import os
import argparse
from ultralytics import YOLO
import cv2
import torch
import torch.nn as nn
import numpy as np
import math
import pandas as pd
import sys  
import datetime


class ChimneyFC(nn.Module):
    def __init__(self):
        super(ChimneyFC, self).__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc3(x)
        return x


def calculate_iou(bbox1, bbox2):
    # 바운딩 박스 좌표 추출
    x1_b1, y1_b1, x2_b1, y2_b1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x1_b2, y1_b2, x2_b2, y2_b2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    # 겹치는 영역 계산
    x_left = max(x1_b1, x1_b2)
    y_top = max(y1_b1, y1_b2)
    x_right = min(x2_b1, x2_b2)
    y_bottom = min(y2_b1, y2_b2)

    if ((x1_b1 < x1_b2 or x1_b1 > x2_b2) and (y1_b1 < y1_b2 or y1_b1 > y2_b2)) and (
            (x2_b1 < x1_b2 or x2_b1 > x2_b2) and (y2_b1 < y1_b2 or y2_b1 > y2_b2)):
        return 0.0  # The bounding boxes do not intersect

    # 겹치는 영역의 넓이 계산
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # 각 바운딩 박스의 넓이 계산
    area_bbox1 = (x2_b1 - x1_b1) * (y2_b1 - y1_b1)
    area_bbox2 = (x2_b2 - x1_b2) * (y2_b2 - y1_b2)

    # IOU 계산
    iou = intersection_area / float(area_bbox1 + area_bbox2 - intersection_area)

    return iou


def line_bbox_intersection(bbox_top_left, bbox_bottom_right, point1, point2):
    # bbox 좌표 분리
    x_min, y_min = bbox_top_left
    x_max, y_max = bbox_bottom_right
    
    # point1, point2 분리
    x1, y1 = point1
    x2, y2 = point2

    # 직선의 기울기와 y 절편 구하기 (y = mx + b)
    if x2 != x1:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
    else:
        m = None  # 수직선의 경우 처리
    
    # bbox 경계선 정의 (상, 하, 좌, 우 경계)
    edges = [
        ((x_min, y_min), (x_max, y_min)),  # 상단 경계 (y = y_min)
        ((x_min, y_max), (x_max, y_max)),  # 하단 경계 (y = y_max)
        ((x_min, y_min), (x_min, y_max)),  # 좌측 경계 (x = x_min)
        ((x_max, y_min), (x_max, y_max))   # 우측 경계 (x = x_max)
    ]

    # 교차점 계산
    intersections = []

    # 상단, 하단 경계와 교차 (y = y_min, y = y_max)
    if m is not None:  # 수직선이 아닌 경우
        for y_edge in [y_min, y_max]:
            x_intersect = (y_edge - b) / m
            if x_min <= x_intersect <= x_max:
                intersections.append((x_intersect, y_edge))

    # 좌측, 우측 경계와 교차 (x = x_min, x = x_max)
    for x_edge in [x_min, x_max]:
        if m is not None:
            y_intersect = m * x_edge + b
        else:  # 수직선의 경우
            y_intersect = None if not (y_min <= y1 <= y_max) else y1

        if y_intersect is not None and y_min <= y_intersect <= y_max:
            intersections.append((x_edge, y_intersect))

    # 교차점 중 2개를 반환
    if len(intersections) > 2:
        intersections = sorted(intersections, key=lambda p: (p[0] - x1)**2 + (p[1] - y1)**2)[:2]

    return intersections


# 로그 출력 함수
def log_and_print(message, newline=True):
    """Prints a message to the terminal and writes it to the log file."""
    global log_file  # log_file이 전역 변수로 정의되어 있어야 합니다.
    if newline:
        message += '\n'
    print(message, end='')  # 터미널 출력
    log_file.write(message)  # 로그 파일에 저장


if __name__ == '__main__':
    # 로그 파일 생성
    log_file_path = './chimney_inference_log.txt'
    log_file = open(log_file_path, 'w')

    # 실행 명령어 기록
    execution_command = ' '.join(sys.argv)
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_and_print(f"Execution Command: {execution_command}")
    log_and_print(f"Start Time: {start_time}\n")

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/pose/weight_yolov8_chimney/weights/chimney.pt', help='model.pt path(s)')
    parser.add_argument('--weights_fc', nargs='+', type=str, default='runs/pose/weight_yolov8_chimney/weights/chimney_fc_best.pth', help='model.pt path(s)')
    parser.add_argument('--img-size', nargs= '+', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', type=str, default='cpu', help='augmented inference')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--save_folder', default='./chimney_evaluate/', type=str, help='Dir to save txt results')
    parser.add_argument('--dataset_folder', default='./data/chimney_yolo/', type=str, help='dataset path')
    opt = parser.parse_args()
    print(opt)

    dist_ratio_issue = 0
    total_instances = 0

    result_dic = {"filename":[], "img_resolution":[], "img_roll_tilt":[], "img_pitch_tilt":[],
                  "t_bbox_lt_x":[], "t_bbox_lt_y":[], "t_bbox_rb_x":[], "t_bbox_rb_y":[], 
                  "t_bbox_kpt_1_x":[], "t_bbox_kpt_1_y":[], "t_bbox_kpt_2_x":[], "t_bbox_kpt_2_y":[], "t_height":[],
                  "p_bbox_lt_x":[], "p_bbox_lt_y":[], "p_bbox_rb_x":[], "p_bbox_rb_y":[],
                  "p_bbox_kpt_1_x":[], "p_bbox_kpt_1_y":[], "p_bbox_kpt_2_x":[], "p_bbox_kpt_2_y":[], "p_height_inf":[], "p_height_cal":[],
                  "sqerr_height_inf":[], "sqerr_height_cal":[]}

    # MAE와 RMSE 계산을 위한 변수들
    sumOfAbsErr_inf = 0      # MAE 계산용 (신경망)
    sumOfAbsErr_cal = 0      # MAE 계산용 (기하학적)
    sumOfSqErr_inf = 0       # RMSE 계산용 (신경망)
    sumOfSqErr_cal = 0       # RMSE 계산용 (기하학적)
    numOfErr = 0
    tp, fp, fn = 0, 0, 0
    all_target_heights = []

    model = YOLO(opt.weights)

    # Chimney_FC MODEL
    model_fc = ChimneyFC()
    model_fc.load_state_dict(torch.load(opt.weights_fc))
    model_fc.eval()

    # testing dataset
    testset_folder = opt.dataset_folder
    testset_list = opt.dataset_folder + "chimney_test.txt"
    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
        num_images = len(test_dataset)
    for img_name in test_dataset:
        image_path = testset_folder + img_name

        img0 = cv2.imread(image_path)  # BGR
        imgWidth, imgHeight, _ = img0.shape

        # chimney_fc data
        target_kpt_list = []
        target_kpts_list = []
        fc_data_path = testset_folder + "test_fc/"+ img_name.split('/')[-1].replace('.jpg', '.txt')
        invalidDataFlag = False
        with open(fc_data_path, 'r') as file:
            # line = file.readline().strip()
            # fc_data_value = [float(x) for x in line.split()]
            lines = file.readlines()
            for line in lines:
                line = line.replace('  ', ' ')
                fc_data_value = [float(x) for x in line.split()]
                if len(fc_data_value) != 8:
                    invalidDataFlag = True
                    break
                
                chn_height = float(fc_data_value[0])
                target_kpt_x1 = float(fc_data_value[1])
                target_kpt_y1 = float(fc_data_value[2])
                target_kpt_x2 = float(fc_data_value[3])
                target_kpt_y2 = float(fc_data_value[4])

                cv2.circle(img0, (int(target_kpt_x1), int(target_kpt_y1)), 2, (255, 0, 0), 2)
                cv2.circle(img0, (int(target_kpt_x2), int(target_kpt_y2)), 2, (255, 0, 0), 2)

                target_kpt_list.append([target_kpt_x1, target_kpt_y1, chn_height])
                target_kpt_list.append([target_kpt_x2, target_kpt_y2, chn_height])
                target_kpts_list.append([target_kpt_x1, target_kpt_y1, target_kpt_x2, target_kpt_y2, chn_height])
        
        if invalidDataFlag == True:
            log_and_print("[ERR] Invalid data---> ", img_name)
            continue       
        
        target_bboxes_list = []
        bbox_data_path = testset_folder + "test/"+ img_name.split('/')[-1].replace('.jpg', '.txt')
        with open(bbox_data_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.replace('  ', ' ')
                bbox_data_value = [float(x) for x in line.split()]

                target_bbox_x1 = (float(bbox_data_value[1]) - float(bbox_data_value[3]) * 0.5) * imgWidth
                target_bbox_y1 = (float(bbox_data_value[2]) - float(bbox_data_value[4]) * 0.5) * imgHeight
                target_bbox_x2 = (float(bbox_data_value[1]) + float(bbox_data_value[3]) * 0.5) * imgWidth
                target_bbox_y2 = (float(bbox_data_value[2]) + float(bbox_data_value[4]) * 0.5) * imgHeight

                target_kpt_x1 = float(bbox_data_value[5]) * imgWidth
                target_kpt_y1 = float(bbox_data_value[6]) * imgHeight
                target_kpt_x2 = float(bbox_data_value[7]) * imgWidth
                target_kpt_y2 = float(bbox_data_value[8]) * imgHeight

                target_bboxes_list.append([target_bbox_x1, target_bbox_y1, target_bbox_x2, target_bbox_y2, target_kpt_x1, target_kpt_y1, target_kpt_x2, target_kpt_y2])

        if len(target_kpts_list) != len(target_bboxes_list):
            log_and_print("[ERR] target size error...!")
            continue

        saveFlag = False

        # inference chimney_yolo
        results = model.predict(source=image_path, imgsz=opt.img_size, conf=opt.conf_thres, iou=opt.iou_thres, augment=opt.augment, device=opt.device, verbose=False)
        
        
        # YOLO 추론 요약 정보 캡처 및 로그 저장
        num_chimneys = len(results[0].boxes)
        yolo_summary = (
            f"image {test_dataset.index(img_name) + 1}/{num_images} {image_path}: {opt.img_size}x{opt.img_size} "
            f"{num_chimneys} chimneys, {results[0].speed['inference']:.1f}ms\n"
            f"Speed: {results[0].speed['preprocess']:.1f}ms preprocess, "
            f"{results[0].speed['inference']:.1f}ms inference, "
            f"{results[0].speed['postprocess']:.1f}ms postprocess per image at shape {results[0].orig_shape}"
        )
        log_and_print("")
        log_and_print(yolo_summary)
        #log_and_print("")  # 한 줄 띄우기

        save_name = opt.save_folder + img_name[:-4] + ".txt"
        save_name_err = opt.save_folder + "/err/" + img_name[:-4] + ".txt"

        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        
        dirname_err = os.path.dirname(save_name_err)
        if not os.path.isdir(dirname_err):
            os.makedirs(dirname_err)

        with open(save_name, "w") as fd:
            result = results[0].cpu().numpy()
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(result.boxes.shape[0]) + '\n'

            for idx, box in enumerate(result.boxes):
                isInlier = False
                doPospProc = False
                minDistKpt1 = 20
                minDistKpt2 = 20
                max_iou = 0
                max_idx = 0
                
                conf = box.conf[0]
                cls  = box.cls[0]
                xyxy = box.xyxy[0]
                x1 = int(xyxy[0] + 0.5)
                y1 = int(xyxy[1] + 0.5)
                x2 = int(xyxy[2] + 0.5)
                y2 = int(xyxy[3] + 0.5)
                
                kpt_xy = result.keypoints[idx].xy[0]
                kpt_x1 = kpt_xy[0][0]
                kpt_y1 = kpt_xy[0][1]
                kpt_x2 = kpt_xy[1][0]
                kpt_y2 = kpt_xy[1][1]

                pred_bbox = [x1, y1, x2, y2]
                pred_kpt_dist = np.sqrt((kpt_x1 - kpt_x2)**2 + (kpt_y1 - kpt_y2)**2)
                pred_bbox_diagonal_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                for idx in range(len(target_bboxes_list)):
                    target_bbox = target_bboxes_list[idx][:4]
                    target_kpt_dist = np.sqrt((target_bboxes_list[idx][4] - target_bboxes_list[idx][6])**2 + (target_bboxes_list[idx][5] - target_bboxes_list[idx][7])**2)

                    iou = calculate_iou(target_bbox, pred_bbox)
                    if iou >= max_iou:
                        max_iou = iou
                        max_idx = idx
                        max_target_kpt_dist = target_kpt_dist

                if max_iou >= 0.5:
                    tp += 1  # True Positive
                    isInlier = True
                    targetChnH = target_kpts_list[max_idx][4]
                    
                    ## key-point 오차 기준으로 error 분류
                    target_kpt_1 = target_bboxes_list[max_idx][4:6]
                    target_kpt_2 = target_bboxes_list[max_idx][6:8]
                    distKpt1ToGt = math.sqrt((kpt_x1 - target_kpt_1[0]) ** 2 + (kpt_y1 - target_kpt_1[1]) ** 2)
                    distKpt2ToGt = math.sqrt((kpt_x2 - target_kpt_2[0]) ** 2 + (kpt_y2 - target_kpt_2[1]) ** 2)

                    if distKpt1ToGt < minDistKpt1:
                        minDistKpt1 = distKpt1ToGt
                    if distKpt2ToGt < minDistKpt2:
                        minDistKpt2 = distKpt2ToGt
                    
                    if (minDistKpt1 > 5) or (minDistKpt2 > 5):
                        saveFlag = True
                    
                    # gt와 pred 의 key-point 간 길이 비율 기준으로 error 분류
                    err_dist_ratio = pred_kpt_dist / max_target_kpt_dist
                    if err_dist_ratio < 0.5:
                        saveFlag = True

                    ## key-point 와 bbox 크기 기준으로 후처리 적용 유무 결정
                    dist_ratio = pred_kpt_dist / pred_bbox_diagonal_dist
                    if dist_ratio < 0.5:
                        dist_ratio_issue += 1
                        doPospProc = True

                        # 보정 로직 적용
                        if True:
                            bbox_top_left = (x1, y1)
                            bbox_bottom_right = (x2, y2)
                            point1 = (kpt_x1, kpt_y1)
                            point2 = (kpt_x2, kpt_y2)
                            inter_point = line_bbox_intersection(bbox_top_left, bbox_bottom_right, point1, point2)
                            
                            # 보정 전 keypoint
                            cv2.circle(img0, (int(kpt_x1), int(kpt_y1)), 2, (0, 255, 0), 1)
                            cv2.circle(img0, (int(kpt_x2), int(kpt_y2)), 2, (0, 255, 0), 1)

                            kpt_x1 = inter_point[0][0]
                            kpt_y1 = inter_point[0][1]
                            kpt_x2 = inter_point[1][0]
                            kpt_y2 = inter_point[1][1]
                    
                    total_instances += 1  
                else:
                    fp += 1  # False Positive
                fd.write('%d %d %d %d %.03f' % (x1, y1, x2-x1, y2-y1, conf if conf <= 1 else 1))
                fd.write(' %f %f %f %f' % (kpt_x1, kpt_y1, kpt_x2, kpt_y2))

                cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.circle(img0, (int(kpt_x1), int(kpt_y1)), 2, (0, 0, 255), 1)
                cv2.circle(img0, (int(kpt_x2), int(kpt_y2)), 2, (0, 0, 255), 1)

                chi_angle = round(math.atan2(kpt_x2 - kpt_x1, kpt_y2 - kpt_y1), 4)

                H_hat_test = np.ones((1, 1))
                H_hat_test[:, 0] = np.sqrt((kpt_x2 - kpt_x1) * (kpt_x2 - kpt_x1) + (kpt_y2 - kpt_y1) * (kpt_y2 - kpt_y1)) * fc_data_value[5]
                denominator_test = np.ones((1, 1))
                denominator_test[:, 0] = np.sqrt(np.tan(np.radians(fc_data_value[6])) * np.tan(np.radians(fc_data_value[6])) + np.tan(np.radians(fc_data_value[7])) * np.tan(np.radians(fc_data_value[7])))
                calculated_chimney_length = H_hat_test / denominator_test

                ## regression_net input version #1
                chimney_fc_input = np.ones((1, 1))
                chimney_fc_input = calculated_chimney_length

                chimney_fc_input = torch.tensor(chimney_fc_input, dtype=torch.float32)
                with torch.no_grad():
                    predictions = model_fc(chimney_fc_input)

                chimney_length = float(predictions.numpy())
                fd.write(' %f %f' % (round(float(chimney_fc_input[0]), 2), round(chimney_length, 2)) + '\n')

                tl = 1 or round(0.002 * (1025) / 2) + 1  # line/font thickness
                tf = max(tl - 1, 1)  # font thickness
                chimney_length_str = str(round(chimney_length, 2)) + " m"
                dist_ratio_str = str(round(dist_ratio, 2))
                cv2.putText(img0, chimney_length_str, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                cv2.putText(img0, dist_ratio_str, (x1, y1 - 2 + 15), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

                if isInlier == True:
                    all_target_heights.append(targetChnH)
                    
                    # MAE 계산 (절댓값 오차)
                    abs_err_inf = abs(chimney_length - targetChnH)
                    abs_err_cal = abs(calculated_chimney_length[0][0] - targetChnH)
                    sumOfAbsErr_inf += abs_err_inf
                    sumOfAbsErr_cal += abs_err_cal
                    
                    # RMSE 계산 (제곱 오차)
                    sq_err_inf = (chimney_length - targetChnH) ** 2
                    sq_err_cal = (calculated_chimney_length[0][0] - targetChnH) ** 2
                    sumOfSqErr_inf += sq_err_inf
                    sumOfSqErr_cal += sq_err_cal
                    
                    numOfErr += 1

                # 추론된 높이 계산
                pred_height = chimney_length
                target_height = targetChnH
                abs_error = abs(target_height - pred_height)  # 절대 오차 계산

                # 로그 출력 추가
                log_and_print(f"Image: {img_name}, Box Top-Left: ({x1}, {y1}), Width: {x2-x1}, Height: {y2-y1}, IoU: {max_iou:.3f}")
                log_and_print(f"GT Height: {target_height:.2f}, Predicted Height: {pred_height:.2f}, AbsError: {abs_error:.2f}")

            cv2.imwrite(save_name.replace('.txt', ".jpg"), img0)

            if saveFlag == True:
                cv2.imwrite(save_name_err.replace('.txt', ".jpg"), img0)

    # 전체 결과 계산
    avg_gt_height = sum(all_target_heights) / len(all_target_heights) if all_target_heights else 0
    
    # MAE 계산
    mae_inf = sumOfAbsErr_inf / numOfErr if numOfErr > 0 else 0
    mae_cal = sumOfAbsErr_cal / numOfErr if numOfErr > 0 else 0
    mae_percentage = (mae_inf / avg_gt_height) * 100 if avg_gt_height > 0 else 0
    
    # RMSE 계산
    rmse_inf = math.sqrt(sumOfSqErr_inf / numOfErr) if numOfErr > 0 else 0
    rmse_cal = math.sqrt(sumOfSqErr_cal / numOfErr) if numOfErr > 0 else 0
    rmse_percentage = (rmse_inf / avg_gt_height) * 100 if avg_gt_height > 0 else 0

    log_and_print("\n=========================================================")
    log_and_print("===== Chimney detection & height estimation RESULT ======")
    log_and_print("=========================================================\n")
    log_and_print(f"[mAP] {tp / (tp + fp) if (tp + fp) > 0 else 0:.4f}")
    log_and_print(f"[MAE-INF] {mae_inf:.4f} m")
    log_and_print(f"[MAE-CAL] {mae_cal:.4f} m")
    log_and_print(f"[RMSE-INF] {rmse_inf:.4f} m")
    log_and_print(f"[RMSE-CAL] {rmse_cal:.4f} m")
    log_and_print(f"[Average GT Height] {avg_gt_height:.2f} m")
    log_and_print(f"[MAE Percentage] {mae_percentage:.2f}%")
    log_and_print(f"[RMSE Percentage] {rmse_percentage:.2f}%")
    
    # 추가 분석 정보
    if rmse_inf > mae_inf:
        ratio = rmse_inf / mae_inf
        log_and_print(f"[Analysis] RMSE/MAE ratio: {ratio:.2f} (>1.0 indicates presence of outliers)")
    
    # 종료 시간 기록
    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_and_print(f"\nEnd Time: {end_time}")
    log_file.close()
