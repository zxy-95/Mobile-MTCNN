#coding:utf-8
import tensorflow as tf
import numpy as np
import os
import sys
rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(0, rootPath)
from training.mtcnn_model import P_Net, R_Net, O_Net
from tools.loader import TestLoader
from detection.MtcnnDetector import MtcnnDetector
from detection.detector import Detector
from detection.fcn_detector import FcnDetector
import cv2
import argparse
import time

def test(stage):
    detectors = [None, None, None]
    if stage in ['pnet', 'rnet', 'onet']:
        modelPath = os.path.join(rootPath, 'tmp/model/pnet/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('pnet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a)) # auto match a max epoch model
        modelPath = os.path.join(modelPath, "pnet-%d"%(maxEpoch))
        print("Use PNet model: %s"%(modelPath))
        detectors[0] = FcnDetector(P_Net,modelPath)
    if stage in ['rnet', 'onet']:
        modelPath = os.path.join(rootPath, 'tmp/rnet/model/middle/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('rnet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "rnet-%d"%(maxEpoch))
        print("Use RNet model: %s"%(modelPath))
        detectors[1] = Detector(R_Net, 24, 1, modelPath)
    if stage in ['onet']:
        modelPath = os.path.join(rootPath, 'tmp/onet/model/small/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('onet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "onet-%d"%(maxEpoch))
        print("Use ONet model: %s"%(modelPath))
        detectors[2] = Detector(O_Net, 48, 1, modelPath)
    mtcnnDetector = MtcnnDetector(detectors=detectors, min_face_size =50, threshold=[0.8, 0.8, 0.8],scale_factor=0.7)

    cap = cv2.VideoCapture(0)
    while(True):
        testImages = []
        ret, image = cap.read()
        image=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        testImages.append(image)
        # Now to detect
        starttime=time.time()
        allBoxes, allLandmarks = mtcnnDetector.detect_face(testImages)
        # print("\n")
        # Save it
        # print(time.time()-starttime)
        for bbox in allBoxes[0]:
            cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
            cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))
        allLandmark = allLandmarks[0]
        if allLandmark is not None: # pnet and rnet will be ignore landmark
            for landmark in allLandmark:
                for i in range(int(len(landmark)/2)):
                    cv2.circle(image, (int(landmark[2*i]),int(int(landmark[2*i+1]))), 3, (0,0,255))
        cv2.imshow("test", image)
        c = cv2.waitKey(1) & 0xFF
        if c == 27 or c == ord('q'):
            break

def parse_args():
    parser = argparse.ArgumentParser(description='Create hard bbox sample...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stage', dest='stage', help='working stage, can be pnet, rnet, onet',
                        default='onet', type=str)
    parser.add_argument('--gpus', dest='gpus', help='specify gpu to run. eg: --gpus=0,1',
                        default='0', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    stage = args.stage
    if stage not in ['pnet', 'rnet', 'onet']:
        raise Exception("Please specify stage by --stage=pnet or rnet or onet")
    # Support stage: pnet, rnet, onet
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus # set GPU
    test(stage)

