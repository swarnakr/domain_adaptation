import os
import sys
sys.path.append(os.getcwd())
import argparse
from model.deeplab import *
from tqdm import tqdm
import json
from utils.metrics import Evaluator
from data import spacenet
from common import config
import numpy as np
import cv2
import pdb
import pickle
import matplotlib.pyplot as plt
import re


def loadBN(seed, sample_number, L):
    search_path = os.path.join('train_log_original_{}'.format(seed),'test_images', 'Vegas_bn')
    all_results = os.listdir(search_path)
    source_results = [file_1 for file_1 in all_results if file_1.endswith('.pth.pickle')]
    all_results = [file_1 for file_1 in all_results if file_1.find('_{}_'.format(sample_number)) != -1]

    
    mean_layerwise_t=dict()
    std_layerwise_t = dict()
    mean_layerwise_s=dict()
    std_layerwise_s = dict()

    for l in range(L):
        mean_layerwise_t[l]=[]
        std_layerwise_t[l]=[]
        mean_layerwise_s[l]=[]
        std_layerwise_s[l]=[]


    bn_source = pickle.load(open(os.path.join(search_path, source_results[0]),'rb'));
    for layer in range(L):
        mean_layerwise_s[layer].append(bn_source[(layer,'mean')]) #each entry in this list contains num_channel values. Total number of list entries equals the number of samples, which is not in any order.
        std_layerwise_s[layer].append(np.sqrt(bn_source[(layer,'var')]))

        
    for bn_path in all_results:
        matches = re.finditer('_',bn_path)
        ind = [match.start() for match in matches]
        if len(ind)<5:
            print(bn_path)
            continue
        
        layer = int(bn_path[ind[3]+1:ind[4]])-1
        dot = bn_path.index('.')
        trial = int(bn_path[ind[4]+1:dot])
        
        bn = pickle.load(open(os.path.join(search_path, bn_path),'rb'));
        mean_layerwise_t[layer].append(bn[(layer,'mean')]) #each entry in this list contains num_channel values. Total number of list entries equals the number of samples, which is not in any order.
        std_layerwise_t[layer].append(np.sqrt(bn[(layer,'var')]))


        
    return mean_layerwise_s, std_layerwise_s, mean_layerwise_t, std_layerwise_t


def plot_params():
    seed = 1
    sample_number = 256
    L = 60
   
    mean_layerwise_s, std_layerwise_s, mean_layerwise_t, std_layerwise_t = loadBN(seed, sample_number, L)
    pdb.set_trace()
    
    aa = np.array(mean_layerwise)
    aa = mean(aa,arr=1)
    #get std
    #compute theta and plot
    
    
    
if __name__ == '__main__':
    plot_params()


