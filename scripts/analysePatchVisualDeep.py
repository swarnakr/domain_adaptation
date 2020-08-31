import os
import sys

# sys.path.append(os.getcwd())
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
from sklearn import metrics
from sklearn.datasets import make_blobs
from yellowbrick.cluster import InterclusterDistance
import seaborn as sns
from sklearn.manifold import TSNE
from PIL import Image

def plot_figure(K,metric_value,save_path,metric_value_name,title_name='elbow method for optimal k',xlabel_name='k'):
    plt.figure()
    plt.plot(K, metric_value, 'bx-')
    plt.xlabel(xlabel_name)
    plt.ylabel(metric_value_name)
    plt.title(title_name)
    #plt.show()
    plt.savefig('{}_{}_{}.png'.format(save_path,metric_value_name,np.max(K)))
    plt.close()
    return

def cluster_metrics(A, labels):
    # intra-cluster distances: ssd of samples to the nearest cluster centre
    silhoutte_score = metrics.silhouette_score(A, labels, metric='euclidean')
    calinski_score = metrics.calinski_harabasz_score(A, labels)
    db_score = metrics.davies_bouldin_score(A, labels)

#    mydict = dict_cluster(i_patches, a_patches, g_patches, city_names,k_means)
#    save_path_k = '{}_{}'.format(save_path,k)
#    gt_ratio = gt_metric(mydict, save_path_k)

    return silhoutte_score, calinski_score, db_score

def gt_metric(mydict,save_path):
    num_labels = len(mydict)
    patch_size = len(mydict[0][1][1].reshape(-1))
    gt_ratio = []
    for label in range(num_labels):
        count_zeros = [len(np.flatnonzero(z[1])) for z in mydict[label]]
        sum_count_zeros = sum(count_zeros)
        sum_all_pix = len(count_zeros)*patch_size
        gt_ratio.append(sum_count_zeros / sum_all_pix)
    plot_figure(range(num_labels), gt_ratio, save_path, 'gt_Ratio', 'GT Ratio for different clusters',
                    'cluster_id')

    return gt_ratio



def vis_square(data,process=1):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    if len(data.shape) == 3:
        pass
    elif len(data.shape) == 4 and data.shape[3] != 3:
        data = np.transpose(data, (0, 2, 3, 1))
    if process:
        mean=np.array([0.3441, 0.3809, 0.4014]); std=np.array([0.1883, 0.2039, 0.2119])
        #for b in range(data.shape[0]):
        #pdb.set_trace()
        data = ((data*std + mean)*255).astype(np.uint8)
    for b in range(data.shape[0]):
        data[b,:] = Image.fromarray(data[b,:])
    # normalize data for display
    #    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    # fig = plt.figure()
    # fig.savefig('./train_log/figs/' + city+ '.png')
    # plt.imshow(data); plt.axis('off')

    return data



def closestviz(mydict, N=100, label=0, city=None):
    assert label in list(range(10))
    if city is None:
        image_list = [z[0] for z in sorted(mydict[label], key=lambda h: h[3])]
    else:
        image_list = [z[0] for z in sorted(mydict[label], key=lambda h: h[3]) if z[2] == city]
        gt_list = [z[1] for z in sorted(mydict[label], key=lambda h: h[3]) if z[2] == city]
    n_min = np.minimum(N, len(image_list))
    if n_min > 0:
        return vis_square(np.array(image_list[:n_min])), vis_square(np.array(gt_list[:n_min]))
    else:
        return


def plots_cluster_patches(get_plots=True, get_cluster_patches=False,good_k=0):
    if get_plots:
        i_patches, a_patches, g_patches, city_names, g_indices = choose_random_subset()
        K = range(2, 10)
        save_path = '/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/ganDA/train_log/figsDA/'
        cluster_metrics(i_patches, a_patches, g_patches, city_names, K, save_path,g_indices)


    if get_cluster_patches:
        # From the above plots and use a good cluster size K
        i_patches, a_patches, g_patches, city_names, g_indices = choose_random_subset() #once the cluster size K is fixed, it shouldn't matter if we choose a different random patches
        model, k_means, A = get_kmeans144_result(a_patches, good_k)
        mydict = dict_cluster(i_patches, a_patches, g_patches, city_names, k_means)

        for label in range(good_k):
            for city in ['Vegas', 'Shanghai']:
                image_disp, gt_disp = closestviz(mydict, 100, label, city, True)
                np.save(
                    '/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/ganDA/train_log/figsDA/closestviz_N_100_label_{}_city_{}_{}_small500'.format(
                        label, city, 'im'), image_disp)
                np.save(
                    '/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/ganDA/train_log/figsDA/closestviz_N_100_label_{}_city_{}_{}_small500'.format(
                        label, city, 'gt'), gt_disp)

            #np.save('/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/ganDA/train_log/figsDA/randomviz_N_100_label_{}_city_{}_144'.format(label,city),randomviz(100,label,city,True))

    return


if __name__ == "__main__":
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--bn', default=False, type=str2bool,
    #         help='whether to use BN adaptation')
    # parser.add_argument('--save_batch', default=0, type=int,
    #         help='number of test images to save (n*batch_size)')
    # parser.add_argument('--cuda', default=True,
    #         help='whether to use GPU')
    # parser.add_argument('--save_path', default='train_log/feats/',
    #         help='path to save images')
    # args = parser.parse_args()
    # test = Test(config, args.bn, args.save_path, args.save_batch, args.cuda)
    # test.test()


