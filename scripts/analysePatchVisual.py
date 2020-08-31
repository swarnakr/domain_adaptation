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

def getImagesActivations():
    images = []
    activations = []
    gts = []
    cities = []
    city_count = []

    target = ['Vegas', 'Shanghai']
    save_path = '/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/ganDA/train_log/feats_layer_deeper' #feats/'
    for city in target:
        train = spacenet.Spacenet(city=city, split='train', img_root='/usr/xtmp/satellite/spacenet/')
        tbar = DataLoader(train, batch_size=16, shuffle=False, num_workers=2)
        cc = 0
        ct = 0
        for i, sample in enumerate(tbar):
            if True:
                image, gt = sample['image'], sample['label']
                activation = torch.load(
                    os.path.join(save_path, "{}Feats_{}.pt".format(city[0], i)))  # map_location=torch.device('cpu')))
                assert image.shape[0] == activation.shape[0]
                for kk in range(image.shape[0]):
                    images.append(image[kk].cpu().numpy())
                    activations.append(activation[kk].cpu().numpy())
                    gts.append(gt[kk].cpu().numpy())
            cc += 1
        city_count.append(len(images))
        cities += [city] * city_count[ct]
        ct += 1

    return images, activations, gts, cities


def getPatches(images, activations, gts, cities):
    N = 10
    assert images[0].shape == (3, 400, 400)
#    assert activations[0].shape == (144, 50, 50)
    image_stride = 40  # 400/N
    activation_stride = 2.5  # 50/N

    i_patches = []
    a_patches = []
    g_patches = []
    c_patches = []
    g_indices = []

    def mynonzero(arr, threshold=0.7, above=True):
        assert threshold <= 1.0 and threshold >= 0.0
        arr2 = arr.reshape(-1)
        if above:
            return len(arr2[arr2 != 0.0]) > threshold * len(arr2)
        else:
            return len(arr2[arr2 != 0.0]) < threshold * len(arr2)

    for image, activation, gt, city in zip(images, activations, gts, cities):
        # for x in [float(f)/2. if f%2==0 else float(f)/2.-0.1 for f in range(2*N-1)]:
        #     for y in [float(g)/2. if g%2==0 else float(g)/2.-0.1 for g in range(2*N-1)]:
        if True: #city == 'Vegas':
            for x in range(N):
                for y in range(N):

                    image_slicer = (slice(None), slice(int(x * image_stride), int((x + 1) * image_stride)),
                                    slice(int(y * image_stride), int((y + 1) * image_stride)))
                    activation_slicer = (slice(None), slice(int(x * activation_stride), int(x * activation_stride + 2)),
                    slice(int(y * activation_stride), int(y * activation_stride+2)))
                    gt_slicer = (slice(int(x * image_stride), int((x + 1) * image_stride)),
                                 slice(int(y * image_stride), int((y + 1) * image_stride)))

                    i_patch = image[image_slicer]
                    a_patch = activation[activation_slicer]
                    g_patch = gt[gt_slicer]
                    #pdb.set_trace()

                    cond1 = mynonzero(i_patch, threshold=0.7, above=True)
                    cond2 = mynonzero(g_patch, threshold=0.7, above=True)
                    cond3 = mynonzero(g_patch, threshold=0.3, above=False)

                    if cond1: # and (cond2 or cond3):
                        i_patches.append(i_patch)
                        a_patches.append(a_patch)
                        g_patches.append(g_patch)
                        c_patches.append(city)
                        g_indices.append(cond2)
                    # else:
                    # pdb.set_trace()
    return i_patches, a_patches, g_patches, c_patches, g_indices


i_a_filename = '/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/ganDA/train_log/i_a_20200406_part_{}_of_5.pickle'
i_a_sample = i_a_filename.format(1)
if os.path.exists(i_a_sample):
    images, activations, gts, cities = [], [], [], []
    for i in range(5):
        d = pickle.load(open(i_a_filename.format(i + 1), 'rb'))
        images = images + d[0]
        activations = activations + d[1]
        gts = gts + d[2]
        cities = cities + d[3]
        del d

else:
    images, activations, gts, cities = getImagesActivations()
    L = len(images)
    chunks = np.array_split(np.arange(L), 5)
    for i, chunk in enumerate(chunks):
        start = chunk[0]
        end = chunk[-1] + 1
        pickle.dump((images[start:end + 1],
                     activations[start:end + 1],
                     gts[start:end + 1],
                     cities[start:end + 1]), open(i_a_filename.format(i + 1), 'wb'))

i_patches, a_patches, g_patches, c_patches, g_indices = getPatches(images, activations, gts,
                                                                   cities)  # getPatches(*getImagesActivations



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

def cluster_metrics(i_patches, a_patches, g_patches, city_names, K, save_path, g_indices):
    # intra-cluster distances: ssd of samples to the nearest cluster centre
    sum_of_squared_distances = []
    silhouette_scores=[]
    calinski_harabasz_scores =[]
    davies_bouldin_scores = []
    k_mean_list=[]
    for k in K:
        model, k_means, A = get_kmeans144_result(a_patches, k)
        k_mean_list.append(k_means)
        sum_of_squared_distances.append(k_means.inertia_)

        labels = k_means.labels_
        score = metrics.silhouette_score(A, labels, metric='euclidean')
        silhouette_scores.append(score)

        score = metrics.calinski_harabasz_score(A, labels)
        calinski_harabasz_scores.append(score)

        score = metrics.davies_bouldin_score(A, labels)
        davies_bouldin_scores.append(score)

        mydict = dict_cluster(i_patches, a_patches, g_patches, city_names,k_means)
        save_path_k = '{}_{}'.format(save_path,k)
        gt_ratio = gt_metric(mydict, save_path_k)

    plot_figure(K, sum_of_squared_distances, save_path, 'sum_of_squared_distances')
    plot_figure(K, silhouette_scores, save_path, 'silhouette_scores')
    plot_figure(K, calinski_harabasz_scores, save_path, 'calinski_harabasz_scores')
    plot_figure(K, davies_bouldin_scores, save_path, 'davies_bouldin_score')

    ssd_best_index = sum_of_squared_distances.index(max(sum_of_squared_distances))
    sil_best_index = silhouette_scores.index(max(silhouette_scores))
    ch_best_index = calinski_harabasz_scores.index(max(calinski_harabasz_scores))
    db_best_index = davies_bouldin_scores.index(max(davies_bouldin_scores))
    #gtr_best_index = gt_ratio.index(max(gt_ratio))

    all_indices = [ssd_best_index, sil_best_index, ch_best_index, db_best_index] #, gtr_best_index] #, axis=None)
    best_k = np.array(K)[np.unique(all_indices)]

    for ind in range(len(K)): #best_k:
      # Visualize output clusters of K means in 2D
      k_means = k_mean_list[ind]
      visualizer = InterclusterDistance(k_means)
      visualizer.fit(A)  # Fit the data to the visualizer
      #visualizer.show()  # Finalize and render the figure
      visualizer.show(outpath='{}_{}_InterclusterDistance.png'.format(save_path,ind))
      visualizer.poof() 

      # Visualize through TSNE
    A_embedded = TSNE().fit_transform(A)
    plt.figure()
    palette = sns.color_palette("bright", 2)
    y_ = np.asarray(g_indices)
    y = y_.astype(np.float32)
    sns.scatterplot(A_embedded[:, 0], A_embedded[:, 1], hue=y, legend='full', palette=palette)
    plt.savefig('{}_tsne.png'.format(save_path))

    return

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


def get_kmeans144_result(a_patches, k):
    # global a_patches
    # filename = '/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/ganDA/train_log/kmeans144_Sh_20200401.pickle'
    # if os.path.exists(filename):
    #    return pickle.load(open(filename, 'rb'))
    # a_patches = a_patches[::2]
    A = np.array([apatch[:, 1, 1].reshape(-1) for apatch in a_patches])
    k_means = KMeans(n_clusters=k, random_state=0).fit(A)
    model = k_means.fit(A)

    # pickle.dump(ans, open(filename, 'wb'))
    return model, k_means, A

def choose_random_subset():
    global i_patches
    global a_patches
    global g_patches
    global c_patches
    global g_indices
    choice = np.random.choice(len(i_patches), 5000, replace=False)

    i_patches2 = [i_patches[ii] for ii in choice]
    a_patches2 = [a_patches[ii] for ii in choice]
    g_patches2 = [g_patches[ii] for ii in choice]
    city_names2 = [c_patches[ii] for ii in choice]
    g_indices2 = [g_indices[ii] for ii in choice]

    return i_patches2, a_patches2, g_patches2, city_names2, g_indices2

def dict_cluster(i_patches, a_patches, g_patches, city_names,res):
    # i_patches,a_patches = getPatches(*getImagesActivations())

    #res, _,_ = get_kmeans144_result(a_patches2, k)
    mydict = dict()
    for ipatch, apatch, gpatch, label, city_name in zip(i_patches, a_patches, g_patches, res.labels_, city_names):
        center = res.cluster_centers_[label]
        if label in mydict.keys():
            mydict[label].append((ipatch, gpatch, city_name, np.linalg.norm(center - apatch[:, 1, 1].reshape(-1))))
        else:
            mydict[label] = [(ipatch, gpatch, city_name, np.linalg.norm(center - apatch[:, 1, 1].reshape(-1)))]

    return mydict


def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    if len(data.shape) == 3:
        pass
    elif len(data.shape) == 4 and data.shape[3] != 3:
        data = np.transpose(data, (0, 2, 3, 1))
    # normalize data for display
    #    data = (data - data.min()) / (data.max() - data.min())
    pdb.set_trace()
    mean=(0.3441, 0.3809, 0.4014); std=(0.1883, 0.2039, 0.2119)
    data *= 255.0
    for b in range(data.shape[0]):
       pdb.set_trace()
       data[b,:] += mean
       data[b,:] *= std

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


def randomviz(N=100, label=0, city=None, my144=False):
    assert label in list(range(10))
    mydict = dict_cluster(my144=my144)

    if city is None:
        image_list = [z[0] for z in mydict[label]]
    else:
        image_list = [z[0] for z in mydict[label] if z[1] == city]

    if len(image_list) > N:
        choice = np.random.choice(len(image_list), N, replace=False)
        return vis_square(np.array([image_list[ii] for ii in choice]))
    else:
        return vis_square(np.array(image_list))


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


