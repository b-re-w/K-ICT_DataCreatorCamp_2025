import os
os.environ["OMP_NUM_THREADS"]='1'

from ultralytics import YOLO
# Load a model
# model = YOLO('yolov8s-pose.yaml')  # build a new model from YAML
# model = YOLO('yolov8s-pose.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8-tiny-pose-chimney.yaml')  # build a new model from YAML
model = YOLO('yolov8x-pose-p6-chimney.yaml')  # build a new model from YAML
# model = YOLO('yolov5-pose-chimney.yaml')  # build a new model from YAML


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    # Train the model
    model.train(data='chimney.yaml', epochs=100, imgsz=1024, batch=64, device=[0])
