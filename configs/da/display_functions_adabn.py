
import matplotlib.pyplot as plt
import pickle
import pdb
from sklearn import metrics
import numpy as np
import sys
sys.path.append('/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/scripts/')
from analysePatchVisualDeep import vis_square
import os 
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.transforms import Affine2D


def adabn_plot_detailed(num, net, expt_name, display_str, city, start):

    jet = cm = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=start+num+1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    x = range(0,113,1)
    x_name = 'Adapted Layers'
    y_name='Accuracy'
    title_name='AdaBN upto K layers'
    #plt.figure()


    for i in range(start,start+num):
        modeldir = 'train_log_{}{}'.format(net,i+1)
        print(modeldir)
        #score_path = os.path.join(modeldir,'models','{}_adabn0.pickle'.format(city))
        adabn_path = os.path.join(modeldir,'models','{}_adabn{}.pickle'.format(city,expt_name))
        #pdb.set_trace()
        if os.path.exists(adabn_path):
            
            #scores = pickle.load(open(score_path,'rb'))
            adabn = pickle.load(open(adabn_path,'rb'))
            colorVal = scalarMap.to_rgba(i)
            y=[]

            #pdb.set_trace()
            #no_adabn_score = scores[1][0]
            #y.append(no_adabn_score)
            y.extend(adabn[0]) #extend y.extend(adabn[0]-no_adabn_score)
            diff_len  = len(x)-len(y)
            if diff_len>0:
                add_arr = np.repeat(y[-1:],diff_len)
                y.extend(add_arr)
                    
            y = np.array(y)*100
            
            plt.plot(x, y, display_str, color = colorVal, label='seed {} '.format(i+1))

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title_name)

    plt.legend()
    #plt.savefig(os.path.join(modeldir,'{}_512.png'.format(modeldir))) #.format(save_path,metric_value_name,np.max(K)))

    #xlim((left, right)) 
    #plt.show()


def adabn_plot_errorbar(num, net, expt_name, display_str, city, start, ifdiff):
    jet = cm = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=num)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    x = range(0,112+ifdiff,1)
    x_name = 'Adapted Layers'
    y_name='Accuracy difference'
    title_name='AdaBN up to K'

    y_all = []
    for i in range(start,start+num):
        modeldir = 'train_log_{}{}'.format(net,i+1)
        print(modeldir)
        adabn_path = os.path.join(modeldir,'models','{}_adabn{}.pickle'.format(city,expt_name))
        if os.path.exists(adabn_path):
            adabn = pickle.load(open(adabn_path,'rb'))
            colorVal = scalarMap.to_rgba(i)
            y=[]
            if expt_name == '':
                no_adabn_score = adabn[0][0]
                adabn_comp = adabn[0][1:]
            else:
                adabn_path = os.path.join(modeldir,'models','{}_adabn.pickle'.format(city))
                adabn_upto = pickle.load(open(adabn_path,'rb'))
                no_adabn_score = adabn_upto[0][0]
                adabn_comp = adabn[0]

            if ifdiff==1:
                y.append(0)
                y.extend(np.diff(adabn[0]))
            else:

                y.extend(adabn_comp-no_adabn_score)

            diff_len  = len(x)-len(y)
            if diff_len>1:
                pdb.set_trace()
                add_arr = np.repeat(y[-1:],diff_len)
                y.extend(add_arr)
                    
            y = np.array(y)*100
            y_all.append(y)

    y_all_np = np.array(y_all)
    y_mean = np.mean(y_all_np,axis=0)
    y_err = np.std(y_all_np,axis=0)
    trans = Affine2D().translate(0.2*1, 0) + ax.transData

    pdb.set_trace()
    plt.errorbar(x, y_mean, yerr=y_err, label=display_str, transform=trans)

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title_name)
    plt.legend()


    #xlim((left, right)) 


if __name__=='__main__':

    num=20
    net = 'resnet_'
    source_or_test=0
    stop_val=45
    fig, ax = plt.subplots()
    city='Paris'    

    #adabn_plot_detailed(num, net, '', '*-', city, -1)
    #adabn_plot_detailed(num, net, '_k', '*-',city, -1)
    #adabn_plot_detailed(20, str0,'_k_1','*--',-1)
    if 1:
        adabn_plot_errorbar(num, net, '' , 'Up to layer K', city, 0,0)
        adabn_plot_errorbar(num, net, '_k', 'At layer K', city, -1,0)
        #adabn_plot_errorbar(num, str0,'_k_1','At layers 1 and K', 3)
        plt.savefig(os.path.join('{}{}_uptok_atk.png'.format(net,city))) #.format(save_path,metric_value_name,np.max(K)))

    if 0:
        adabn_plot_errorbar(num, net, '' , 'Up to layer K', city, -1,1)
        plt.savefig(os.path.join('{}{}_uptok_diff.png'.format(net,city)))
    
    plt.show()


    
