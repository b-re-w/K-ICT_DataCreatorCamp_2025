import os
import numpy as np
import cv2
import glob
import json
import math
import shutil
import random

dataDir = "data/chimney_yolo"

if __name__ == "__main__":
    outFilePath = os.path.join(dataDir, "chimney_test.txt")
    imageList = glob.glob(os.path.join(dataDir, 'val', '*.jpg'))
    with open(outFilePath, 'w') as fp:
        for imagePath in imageList:
            fp.write('/'.join(imagePath.split('/')[-2:]) + '\n')
