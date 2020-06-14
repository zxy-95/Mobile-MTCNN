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
from tqdm import tqdm
def test(stage, testFolder):
    print("Start testing in %s"%(testFolder))
    detectors = [None, None, None]
    if stage in ['pnet', 'rnet', 'onet']:
        modelPath = os.path.join(rootPath, 'tmp/origin/model/pnet')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('pnet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a)) # auto match a max epoch model
        modelPath = os.path.join(modelPath, "pnet-%d"%(maxEpoch))
        print("Use PNet model: %s"%(modelPath))
        detectors[0] = FcnDetector(P_Net,modelPath)
    if stage in ['rnet', 'onet']:
        modelPath = os.path.join(rootPath, 'tmp/origin/model/rnet')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('rnet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "rnet-%d"%(maxEpoch))
        print("Use RNet model: %s"%(modelPath))
        detectors[1] = Detector(R_Net, 24, 1, modelPath)
    if stage in ['onet']:
        modelPath = os.path.join(rootPath, 'tmp/origin/model/onet')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('onet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "onet-%d"%(maxEpoch))
        print("Use ONet model: %s"%(modelPath))
        detectors[2] = Detector(O_Net, 48, 1, modelPath)
    mtcnnDetector = MtcnnDetector(detectors=detectors, min_face_size = 50, threshold=[0.8, 0.9, 0.9])



    fileFoldName = "faceListInt.txt"

    outFilename = 'F:/software/yansan/MTCNN-on-FDDB-Dataset-master/FDDB-folds/' + 'predict.txt'  # fileOutName
    foldFilename = 'F:/software/yansan/MTCNN-on-FDDB-Dataset-master/FDDB-folds/' + fileFoldName

    prefixFilename = 'E:/database/FDDB_Face Detection Data Set and Benchmark/'

    fout = open(outFilename, 'a+')


    f = open(foldFilename, 'r')  # FDDB-fold-00.txt, read
    for imgpath in tqdm(f.readlines()):
        testImages = []
        imgpath = imgpath.split('\n')[0]
        # foutOnce.write(imgpath+'\n')
        # foutFold.write(imgpath+'\r')
        img = cv2.imread(prefixFilename + imgpath + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if img is None:
            continue
        testImages.append(img)
        boundingboxes, points = mtcnnDetector.detect_face(testImages)
        # boundingboxes, points = demo.detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)

        text1 = str(imgpath) + '\n' + str(len(boundingboxes[0])) + '\n'
        fout.write(text1)  # FDDB-fold-%02d-out.txt or predict.txt

        for bbox in boundingboxes[0]:
            # print(bbox,"???")
            text2 = str(int(bbox[0])) + ' ' + str(int(bbox[1])) + ' ' \
                    + str(abs(int(bbox[2] - bbox[0]))) + ' ' \
                    + str(abs(int(bbox[3] - bbox[1]))) + ' ' \
                    + str(bbox[4]) + '\n'

            fout.write(text2)  # FDDB-fold-%02d-out.txt or predict.txt
            # text2 = str(int(boundingboxes[coordinate][0][0]))

            # fout.write(text2)  # FDDB-fold-%02d-out.txt or predict.txt

    # print error
    f.close()  # input the fold list, FDDB-fold-00.txt
    fout.close()  # output the result, predict.txt


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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus # set GPU
    test(stage, os.path.join(rootPath, "testing", "images"))