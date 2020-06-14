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

def test(stage, testFolder):
    print("Start testing in %s"%(testFolder))
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
    mtcnnDetector = MtcnnDetector(detectors=detectors, min_face_size =12, threshold=[0.6, 0.6, 0.7],scale_factor=0.7)

    testImages = []
    for name in os.listdir(testFolder):
        testImages.append(os.path.join(testFolder, name))

    print("\n")
    right_num=0
    miss_num=0
    FN=0
    # Save it
    for idx, imagePath in enumerate(testImages):
        if(idx<=6000):
            image = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            print(right_num,FN,miss_num)
            try:
                allBoxes, allLandmarks = mtcnnDetector.detect_face([image])
                if(allBoxes.__len__()==1):
                    right_num+=1
                else:
                    FN+=(allBoxes.__len__()-1)

            except:
                miss_num+=1
                pass
        else:
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
    test(stage, 'E:\database\headtracker_sequences\seq_villains2')

