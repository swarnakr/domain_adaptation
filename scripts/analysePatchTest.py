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
from sklearn import metrics
from sklearn.datasets import make_blobs
#from yellowbrick.cluster import InterclusterDistance


def getImagesActivations():

    images = []
    activations = []
    gts=[]
    cities=[]
    city_count=[] 
 
    target = ['Vegas', 'Shanghai']
    save_path = '/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/ganDA/train_log/feats/'
    for city in target:
        val = spacenet.Spacenet(city=city, split='val', img_root='/usr/xtmp/satellite/spacenet/')
        tbar  = DataLoader(val, batch_size=16, shuffle=False, num_workers=2)
        cc=0 
        ct=0
        for i, sample in enumerate(tbar):
            if True:
              image, gt  = sample['image'], sample['label']
              for kk in range(image.shape[0]):
                  images.append(image[kk].cpu().numpy())
                  gts.append(gt[kk].cpu().numpy())
            cc+=1
        city_count.append(len(images))
        cities += [city] * city_count[ct] 
        ct+=1

    return images, gts, cities

def getPatches(images,gts,cities):
    N = 10
    assert images[0].shape==(3,400,400)

    image_stride = 40 # 400/N
    activation_stride = 5 # 50/N

    i_patches = []
    a_patches = []
    g_patches = []
    c_patches = []
    g_indices = []
    def mynonzero(arr, threshold=0.7, above=True):
        assert threshold <= 1.0 and threshold >=0.0
        arr2 = arr.reshape(-1)
        if above:
            return len(arr2[arr2!=0.0]) > threshold * len(arr2)
        else:
            return len(arr2[arr2!=0.0]) < threshold * len(arr2)
    
    for image, gt, city in zip(images, gts,cities):
        # for x in [float(f)/2. if f%2==0 else float(f)/2.-0.1 for f in range(2*N-1)]:
        #     for y in [float(g)/2. if g%2==0 else float(g)/2.-0.1 for g in range(2*N-1)]:
     if city=='Shanghai':
        for x in range(N):
            for y in range(N):
                
                image_slicer      = (slice(None), slice(int(x*image_stride), int((x+1)*image_stride)), slice(int(y*image_stride), int((y+1)*image_stride)))
                gt_slicer         = (slice(int(x*image_stride), int((x+1)*image_stride)), slice(int(y*image_stride), int((y+1)*image_stride)))
                
                i_patch = image[image_slicer]
                g_patch = gt[gt_slicer]

                cond1 = mynonzero(i_patch, threshold=0.7, above=True)
                cond2 = mynonzero(g_patch, threshold=0.7, above=True)
                cond3 = mynonzero(g_patch, threshold=0.3, above=False)                

                if cond1 and (cond2 or cond3):
                    i_patches.append(i_patch)
                    g_patches.append(g_patch)
                    c_patches.append(city)
                    g_indices.append(cond2)
                #else:
                    #pdb.set_trace()
    return i_patches, g_patches, c_patches, g_indices

i_a_filename = '/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/ganDA/train_log/i_test_20200405_part_{}_of_5.pickle'
i_a_sample = i_a_filename.format(1)

if os.path.exists(i_a_sample):
    images, activations, gts, cities = [], [], [], []
    for i in range(5):
        d = pickle.load(open(i_a_filename.format(i+1), 'rb'))
        images = images + d[0]
        gts = gts + d[1]
        cities = cities + d[2]
        del d
        
else:
    images, gts, cities = getImagesActivations()
    L = len(images)
    chunks = np.array_split(np.arange(L), 5)
    for i, chunk in enumerate(chunks):
        start = chunk[0]
        end = chunk[-1]+1
        pickle.dump((images[start:end+1],
                     gts[start:end+1],
                     cities[start:end+1]), open(i_a_filename.format(i+1), 'wb'))



i_patches, g_patches, c_patches, g_indices = getPatches(images, gts, cities) #getPatches(*getImagesActivations())


# def cluster_metrics(a_patches2, K):
#     #intra-cluster distances
#     sum_of_squared_distances = []
#     K = range(1,15)
#     for k in K:
#         k_means = get_kmeans144_result(a_patches2,k)
#         model = k_means.fit(X)
#         sum_of_squared_distances.append(k_means.inertia_)

#     #for each value of k, we can initialise k_means and use inertia to identify the sum of squared distances of samples to the nearest cluster centre
#     labels = k_means.labels_
#     metrics.silhouette_score(X, labels, metric = 'euclidean')
#     metrics.calinski_harabasz_score(X, labels)
#     metrics.davies_bouldin_score(X, labels)

# Generate synthetic dataset with 12 random clusters
# X, y = make_blobs(n_samples=1000, n_features=12, centers=12, random_state=42)

# # Instantiate the clustering model and visualizer
# model = KMeans(6)
# visualizer = InterclusterDistance(model)

# visualizer.fit(X)        # Fit the data to the visualizer
# visualizer.show()        # Finalize and render the figure

# plt.plot(K, sum_of_squared_distances, 'bx-')
# plt.xlabel('k')
# plt.ylabel('sum_of_squared_distances')
# plt.title('elbow method for optimal k')
# plt.show()

def get_kmeans_result():
    global a_patches
    filename = '/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/ganDA/train_log/kmeans_20200326.pickle'
    if os.path.exists(filename):
        return pickle.load(open(filename, 'rb'))
    a_patches2 = a_patches[::2]
    A = np.array([apatch.reshape(-1) for apatch in a_patches2])
    return KMeans(n_clusters=4, random_state=0).fit(A)

def get_kmeans144_result(a_patches2,k):
    #global a_patches
    #filename = '/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/ganDA/train_log/kmeans144_Sh_20200401.pickle'
    #if os.path.exists(filename):
    #    return pickle.load(open(filename, 'rb'))
    #a_patches2 = a_patches[::2]
    A = np.array([apatch[:,2,2].reshape(-1) for apatch in a_patches2])
    model = k_means.fit(X)
    ans = KMeans(n_clusters=k, random_state=0).fit(A)
    #pickle.dump(ans, open(filename, 'wb'))
    return ans


def dict_cluster(my144=False):
    #i_patches,a_patches = getPatches(*getImagesActivations())
    global i_patches
    global a_patches
    global g_patches
    global c_patches

    choice = np.random.choice(len(i_patches), 500, replace=False)    

    i_patches2 = [i_patches[ii] for ii in choice]
    a_patches2 = [a_patches[ii] for ii in choice]
    g_patches2 = [g_patches[ii] for ii in choice]
    city_names2 = [c_patches[ii] for ii in choice]

    if my144:
        res = get_kmeans144_result(a_patches2,k)
    else:
        res = get_kmeans_result()
    mydict = dict()
    for ipatch, apatch, gpatch, label, city_name in zip(i_patches2, a_patches2, g_patches2, res.labels_, city_names2):
        center = res.cluster_centers_[label]
        if label in mydict.keys():
            mydict[label].append((ipatch, gpatch, city_name, np.linalg.norm(center - apatch[:,2,2].reshape(-1))))
        else:
            mydict[label] = [(ipatch, gpatch, city_name, np.linalg.norm(center - apatch[:,2,2].reshape(-1)))]

    return mydict



def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""


    if len(data.shape)==3:
        pass
    elif len(data.shape)==4 and data.shape[3]!=3:
        data = np.transpose(data, (0,2,3,1))
    # normalize data for display
#    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
   
    #fig = plt.figure()
    #fig.savefig('./train_log/figs/' + city+ '.png') 
    #plt.imshow(data); plt.axis('off')

    return data



def randomviz(N=100, label=0, city=None,my144=False):
    assert label in list(range(10))
    mydict = dict_cluster(my144=my144)

    if city is None:
        image_list = [z[0] for z in mydict[label]]
    else:
        image_list = [z[0] for z in mydict[label] if z[1]==city]
    
    if len(image_list)>N:
        choice = np.random.choice(len(image_list), N, replace=False)
        return vis_square(np.array([image_list[ii] for ii in choice]))
    else:
        return vis_square(np.array(image_list))


def closestviz(mydict,N=100, label=0, city=None, my144=False):
    assert label in list(range(10))
    if city is None:
        image_list = [z[0] for z in sorted(mydict[label], key=lambda h:h[3])]
    else:
        image_list = [z[0] for z in sorted(mydict[label], key=lambda h:h[3]) if z[2]==city]
        gt_list = [z[1] for z in sorted(mydict[label], key=lambda h:h[3]) if z[2]==city]
    NMin = np.minimum(N,len(image_list)) 
    if NMin>0:
      return vis_square(np.array(image_list[:NMin])), vis_square(np.array(gt_list[:NMin]))
    else:
      return

def getAllData():
    mydict = dict_cluster(True,k)
    for label in range(4):
           for city in ['Vegas', 'Shanghai']:
               imageDisp, gtDisp = closestviz(mydict,100,label,city,True)
               np.save('/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/ganDA/train_log/figsDA/closestviz_N_100_label_{}_city_{}_{}_small500'.format(label,city,'im'),imageDisp)
               np.save('/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/ganDA/train_log/figsDA/closestviz_N_100_label_{}_city_{}_{}_small500'.format(label,city,'gt'),gtDisp)


               #np.save('/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/ganDA/train_log/figsDA/randomviz_N_100_label_{}_city_{}_144'.format(label,city),randomviz(100,label,city,True))     

    return

def dispAllData():
    for label in range(4):
           for city in ['Vegas', 'Shanghai']:
               data = np.load('/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/ganDA/train_log/figs/randomviz_N_100_label_{}_city_{}_144.npy'.format(label,city))
               fig = plt.figure(); plt.imshow(data);
               fig.savefig('/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/ganDA/train_log/figs/randomviz_N_100_label_{}_city_{}_144.png'.format(label,city));

               data = np.load('/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/ganDA/train_log/figs/closestviz_N_100_label_{}_city_{}_144.npy'.format(label,city))
               fig = plt.figure(); plt.imshow(data);
               fig.savefig('/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/ganDA/train_log/figs/closestviz_N_100_label_{}_city_{}_144.png'.format(label,city));


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

