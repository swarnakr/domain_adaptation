
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
import matplotlib.colors as colors
import matplotlib.cm as cmx

def load_params(seed, sample_number, L):
    search_path = os.path.join('train_log_original_{}'.format(seed),'test_images', 'Vegas_bn')
    all_results = os.listdir(search_path)


    source_results = [file_1 for file_1 in all_results if file_1.startswith('params')]
    pdb.set_trace()
    weight_layerwise_s=dict()
    bias_layerwise_s = dict()

    for l in range(L):
        weight_layerwise_s[l]=[]
        bias_layerwise_s[l]=[]

    params = pickle.load(open(os.path.join(search_path, source_results[0]),'rb'));
    for layer in range(L):
        weight_layerwise_s[layer].append(params[(layer,'gamma')]) #each entry in this list contains num_channel values. Total number of list entries equals the number of samples, which is not in any order.
        bias_layerwise_s[layer].append(params[(layer,'delta')])

    return weight_layerwise_s, bias_layerwise_s
    

    
def loadBN(seed, sample_number, L):
    search_path = os.path.join('train_log_original_{}'.format(seed),'test_images', 'Vegas_bn')
    all_results = os.listdir(search_path)
    source_results = [file_1 for file_1 in all_results if file_1.endswith('.pth.pickle') and not file_1.startswith('params')]
    pdb.set_trace()
    #all_results = [file_1 for file_1 in all_results if file_1.find('_{}_'.format(sample_number)) != -1]

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

    #pdb.set_trace()
    start = source_results[0].index('epoch')
    stop = source_results[0].index('.pth')
    modelname = source_results[0][start:stop]
        
    for layer in range(L):
        for trial in range(30):
            bn = pickle.load(open(os.path.join(search_path, 'bnAll_after_{}_{}_{}_{}.pickle'.format(modelname,sample_number,layer+1,trial)),'rb'));
            mean_layerwise_t[layer].append(bn[(layer,'mean')]) #each entry in this list contains num_channel values. Total number of list entries equals the number of samples, which is not in any order.
            std_layerwise_t[layer].append(np.sqrt(bn[(layer,'var')]))

    return mean_layerwise_s, std_layerwise_s, mean_layerwise_t, std_layerwise_t


def get_theta_mean(layerwise_t, layerwise_s, layerwise_std_t):
    t_array = np.array(layerwise_t)
    theta = np.abs(t_array - layerwise_s) / layerwise_std_t # / std_t
    return theta

def get_theta_std(layerwise_std_t, layerwise_std_s):
    t_array = np.array(layerwise_std_t)
    theta = np.abs(1 - layerwise_std_s / t_array) # / std_t
    return theta

def plot_params(seed):
    sample_number = 256
    L = 60
    mean_layerwise_s, std_layerwise_s, mean_layerwise_t, std_layerwise_t = loadBN(seed, sample_number, L)
    pickle.dump((mean_layerwise_s, std_layerwise_s, mean_layerwise_t, std_layerwise_t),open('/home/home1/swarnakr/nst/train_log_original_{}/test_images/Vegas_bn/mean_std_s_t.pickle'.format(seed),'wb'))


    #pdb.set_trace()
    #mean_layerwise_s, std_layerwise_s, mean_layerwise_t, std_layerwise_t = pickle.load(open('/home/home1/swarnakr/nst/train_log_original_{}/test_images/Vegas_bn/mean_std_s_t.pickle'.format(seed),'rb'))

    theta_mean=[]
    theta_std=[]
    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    jet = cm = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=30)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    
    for i in range(30):    
        for l in range(L):
            scale = np.abs(np.mean(mean_layerwise_t[l]))
            scale2 = np.abs(np.min(std_layerwise_t[l]))

            theta_m = get_theta_mean(mean_layerwise_t[l][i], mean_layerwise_s[l],std_layerwise_t[l][i])
            theta_s = get_theta_std(std_layerwise_t[l][i], std_layerwise_s[l])
            
            theta_mean.append(theta_m)
            theta_std.append(theta_s)

            colorVal = scalarMap.to_rgba(i)
            
            plt.figure(1)
            plt.plot(l, np.median(theta_m), '*', color=colorVal, label='{} sample '.format(i))
            
            plt.figure(2)
            plt.plot(l, np.median(theta_s), '*', color=colorVal, label='{} sample '.format(i))

    fig1.savefig('means_{}.png'.format(seed))
    fig2.savefig('stds_{}.png'.format(seed))
    plt.show()    


def plot_gamma_delta():
    seed = 0
    sample_number = 256
    L = 60
    weights, biases = load_params(seed, sample_number, L)
    fig1 = plt.figure(1)
    fig2 = plt.figure(2)

    for l in range(L):
        plt.figure(1)

        plt.plot(l, np.array(weights[l]), '*')
        plt.plot(l, np.median(weights[l]), 'o', color='r',markersize=5)
        
        plt.figure(2)
        plt.plot(l, np.array(biases[l]), '*')
        plt.plot(l, np.median(biases[l]), 'o', color='r', markersize=5)

    fig1.savefig('gammas.png')
    fig2.savefig('deltas.png')
    plt.show()
        
if __name__ == '__main__':

    # plot_params()
    #plot_gamma_delta()
    for i in range(0,12):
        plot_params(i)
        break
    

