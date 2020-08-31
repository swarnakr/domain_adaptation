import os
import sys
#sys.path.append(os.getcwd())
sys.path.append('/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/ganDA')
import torch
import argparse
from model.deeplab import *
from tqdm import tqdm
import json
from utils.metrics import Evaluator
from data import spacenet
from common import config
from torch.utils.data import DataLoader
import numpy as np
import cv2
import pdb
from sklearn.cluster import KMeans
import pickle
from matplotlib import pyplot as plt



def dispAllData():
    for label in range(4):
           for city in ['Vegas','Shanghai']:
#               data = np.load('/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/ganDA/train_log/figsDA/randomviz_N_100_label_{}_city_{}_144.npy'.format(label,city))
#               fig = plt.figure(); plt.imshow(data);
#               fig.savefig('/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/ganDA/train_log/figsDA/randomviz_N_100_label_{}_city_{}_144.png'.format(label,city));
#               plt.close(fig)
               for dtype in ['im','gt']:
                   data = np.load('/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/ganDA/train_log/figsDA/closestviz_N_100_label_{}_city_{}_{}_small500.npy'.format(label,city,dtype))
                   data = np.uint8(data)
                   fig = plt.figure(); plt.imshow(data);
                   fig.savefig('/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/ganDA/train_log/figsDA/closestviz_N_100_label_{}_city_{}_{}_small500.png'.format(label,city,dtype));

                   plt.close(fig)
    return

