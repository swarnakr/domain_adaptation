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

def loadBN(city):
    base_dir = './train_log/test_images/'
    load_path = os.path.join(base_dir, city + '_bn', 'bnAll_before.pickle')
    bn = pickle.load(open(load_path,'rb'));
#    maxL = 60;
#    maxIm = len(bn)/maxL;
    return bn#,maxL,maxIm)

def meanNStd(data,key):
    m = []
    am = []
    s = []
    for l in sorted(list(np.unique([z[1] for z in data.keys()]))):
        m.append((data[(key, l)]).mean())
        am.append(np.abs(data[(key, l)]).mean())
        s.append((data[(key, l)]).std())
      
    return (np.array(m),np.array(am),np.array(s))

def compareLayers(city):
    bn = loadBN(city)
    #rnd = np.random.randint(maxIm)
    maxL = int((len(bn)-1)/3) # three keys, last conv doesn't have a corresponding BN layer after it
    data = dict()
    pdb.set_trace()
    for l in range(0,maxL):

         data[('mu', l)] = bn[(l,'mean')] #gives the mean parameter for batch norm at layer l, channel i

         data[('sigma', l)] = np.sqrt(bn[(l,'var')]) #gives the std parameter for batch norm at layer l, channel i

         wt= bn[(l,'weight')]
         ch = wt.shape[0]
         assert ch == bn[l,'mean'].shape[0]
         numParams = np.prod(wt.shape[1:])

         for c in range(ch):
            data[('w', l, c)] = np.sqrt(np.sum(wt[c]**2)/numParams)  #gives the 2-norm of the incoming weights on to layer l, channel i as defined in the latex document.
         data[('w', l)] = np.array([data[('w',l,c)] for c in range(ch)])
         #pdb.set_trace()
    #pickle.dump( d, open( "~/triple_dict.pickle", "wb" ) )

    meanWts = np.array([data[('w',l)].mean() for l in range(maxL)])
    # Regular means 
    m,  am, s = meanNStd(data,'mu')
    ms,ams,ss = meanNStd(data,'sigma')

    # # Normalized means
    for l in range(0,maxL):
       data[('wmu',l)] = data[('mu', l)] / data[('w', l)]
       data[('wsigma',l)] = data[('sigma', l)] / data[('w', l)]

    wm,wam,ws    = meanNStd(data,'wmu')
    wms,wams,wss = meanNStd(data,'wsigma')
    #pdb.set_trace() 
    #plt.figure(); plt.scatter(range(maxL),m); plt.plot(m,'r--') #wms and wams same
    #plt.figure(); plt.scatter(range(maxL),am); plt.plot(am,'r--') #wms and wams same
    if 1:
       fig = plt.figure(); plt.plot(am,'r--'); plt.errorbar(range(maxL),am,yerr=s, fmt='.')
       plt.ylim((-0.1, 1)); plt.xlabel(r'Layer $\ell$'); plt.ylabel(r"Mean of $|\mu_i^\ell|$ over $i$"); #plt.legend()
       pdb.set_trace()
       fig.savefig('./train_log/figs/mean_' + city+ '.png')

       fig = plt.figure(); plt.plot(wm,'r--'); plt.errorbar(range(maxL),wm,yerr=ws, fmt='.')
       plt.ylim((-10, 1.5)); plt.xlabel(r'Layer $\ell$'); plt.ylabel(r"Mean of $\alpha_i^\ell$ over $i$");# plt.legend()  
       fig.savefig('./train_log/figs/weighted_mean_notabs_' + city+ '.png')

    #plt.figure(); plt.scatter(range(maxL),wm); plt.plot(wm,'r--') #wms and wams same
    fig = plt.figure(); plt.plot(wam,'r--'); plt.errorbar(range(maxL),wam,yerr=ws, fmt='.')
    plt.ylim((-2, 12)); plt.xlabel(r'Layer $\ell$'); plt.ylabel(r"Mean of $|\alpha_i^\ell|$ over $i$");# plt.legend()  
    fig.savefig('./train_log/figs/weighted_mean_' + city+ '.png')
    #plt.figure(); plt.plot(s); plt.figure(); plt.plot(ws); plt.show()
    fig = plt.figure(); plt.plot(ms,'r--'); plt.errorbar(range(maxL),ms,yerr=ss, fmt='.') #ms and ams are same for sigma
    plt.ylim((-1, 16));  plt.xlabel(r'Layer $\ell$'); plt.ylabel(r"Mean of $|\sigma_i^\ell|$ over $i$"); #plt.legend()
    fig.savefig('./train_log/figs/mean_std_' + city+ '.png')

    #plt.figure(); plt.scatter(range(maxL),ws); plt.plot(ws,'r--') #wms and wams same
    fig = plt.figure(); plt.plot(wms,'r--'); plt.errorbar(range(maxL),wms,yerr=wss, fmt='.')
    plt.ylim((-1, 16));  plt.xlabel(r'Layer $\ell$'); plt.ylabel(r"Mean of $|\beta_i^\ell|$ over $i$"); #plt.legend()
    #plt.figure(); plt.plot(ss); plt.figure(); plt.plot(wss); plt.show()
    fig.savefig('./train_log/figs/weighted_std_' + city+ '.png')

    return data, [m,am,s,ms,ams,ss,wm,wam,ws,wms,wams,wss]
    
#assert that conv size is correct

if __name__ == '__main__':
#    city = str(sys.argv[1])
  #  compareLayers(city)

    #compareLayers('ShanghaiCol1') 
    #compareLayers('ShanghaiCol2')
    #compareLayers('ShanghaiHeq')

    #compareLayers('VegasCol1') 
    #compareLayers('VegasCol2')
    compareLayers('Vegas')


#    compareCities()
