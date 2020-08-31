
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

def _fpr_tpr(filename,score_invert, pos_label,regression=0):
    myfilename = '/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/cluster_patch/train_log/eval/' + filename 
    (build_label_all,scores) = pickle.load(open(myfilename, 'rb'))
    build_label_all = (build_label_all).reshape(-1)
    build_scores=[]
    if regression:
        build_scores = build_label_all
        build_label_all = np.array(build_label_all) > 0.5
        
    scores_np = np.array(scores)
    if score_invert:
        scores_np = - scores_np

    fpr, tpr, thresholds = metrics.roc_curve(build_label_all,scores_np, pos_label=pos_label
    )
    return fpr,tpr,thresholds, scores_np, build_scores
    
def plot_roc(regression):
    if 0:
        fpr_sa,tpr_sa,thresholds_sa, _,_ = _fpr_tpr('roc_k_source_adabn_2.pickle',1,0)
        fpr_s,tpr_s,thresholds_s, _,_ = _fpr_tpr('roc_k_source.pickle',1,0)
        fpr_i,tpr_i,thresholds_i, _,_ = _fpr_tpr('roc_k_imagenet.pickle',1,0)
        fpr_sa_1,tpr_sa_1,thresholds_sa_1, scores,_ = _fpr_tpr('roc_k_source_adabn_2.pickle',0,1)

        fpr_sa,tpr_sa,thresholds_sa, _, _ = _fpr_tpr('regression_target_adabn.pickle',1,0,regression)
        fpr_sa_1,tpr_sa_1,thresholds_sa_1, scores, build_scores = _fpr_tpr('regression_target_adabn.pickle',0,1,regression)

    fpr_sa,tpr_sa,thresholds_sa, _, _ = _fpr_tpr('cluster_unsupervised_source.pickle',1,0,regression)
    fpr_sa_1,tpr_sa_1,thresholds_sa_1, scores, build_scores = _fpr_tpr('cluster_unsupervised_target.pickle',0,1,regression)
    if 0:
        num_values = len(scores)
        min_length = np.minimum(tpr_sa.shape[0], tpr_sa_1.shape[0])
        ratios = ((tpr_sa[:min_length] + tpr_sa_1[:min_length])/2) / ((fpr_sa[:min_length] + fpr_sa_1[:min_length])/2) 
        count_correct = ((tpr_sa[:min_length] *(num_values/2)) + (tpr_sa_1[:min_length] *(num_values/2))) / 2
        inds = np.nonzero((ratios>2.9) & (count_correct > 200) & (abs(thresholds_sa_1 - thresholds_sa)<0.005)
        )
    #ind_choose = np.argmax(count_correct[inds])
    #threshold_chosen = (thresholds_sa[inds])[ind_choose]
    
    if 1:
        plt.figure()
        lw = 2
        plt.plot(fpr_sa, tpr_sa, color='darkorange',lw=2, label='Train with Source and AdaBN Class 1')
        plt.plot(fpr_sa_1, tpr_sa_1, color='darkgreen',lw=2, label='Train with Source and AdaBN Class 2')
        #plt.plot(fpr_s, tpr_s, color='darkgreen',lw=2, label='Train with Source')
        #plt.plot(fpr_i, tpr_i, color='darkred',lw=2, label='Train with Imagenet')    
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")

    pdb.set_trace()
    if regression:
        plt.figure()
        lw = 2
        build_scores = np.array(build_scores)
        bottom_left = ((build_scores<=0.5) & (scores<=500)).sum() / float(len(scores)) *100
        bottom_right = ((build_scores>0.5) & (scores<=500)).sum() / float(len(scores)) *100
        top_right = ((build_scores>0.5) & (scores>=500)).sum() / float(len(scores)) *100
        top_left = ((build_scores<=0.5) & (scores>=500)).sum() / float(len(scores)) *100

        plt.scatter(build_scores, scores, color='darkred',marker='.',alpha=0.5)
        
        plt.xlim([0.0, 1.05])
        plt.ylim([-1.05, 1.05])
        plt.text(0.2,-0.4, "{:.2f}%".format(bottom_left), fontsize=20, fontweight='bold')
        plt.text(0.7,-0.4, "{:.2f}%".format(bottom_right), fontsize=20, fontweight='bold')
        plt.text(0.2,0.4, "{:.2f}%".format(top_left), fontsize=20, fontweight='bold')
        plt.text(0.7,0.4, "{:.2f}%".format(top_right), fontsize=20, fontweight='bold')        
        
        plt.xlabel('Groundtruth Building Score')
        plt.ylabel('K means Distance Score')
        plt.title('Correlation of Building Scores')
                                                                                                    
    plt.show()
    pdb.set_trace()
    return #threshold_chosen



def plot_visualization_fp():
        myfilename = '/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/cluster_patch/train_log/eval/fp_tp_patch.pickle'
        (false_positives_B, false_positives_NB, true_positives_B, true_positives_NB, _,_,_,_) = pickle.load(open(myfilename, 'rb'))

        data_B = vis_square(np.array(false_positives_B[:64]))
        data_NB = vis_square(np.array(false_positives_NB[:64]))
        fig = plt.figure(); plt.imshow(data_B); plt.grid(b=None)
        fig = plt.figure(); plt.imshow(data_NB); plt.grid(b=None)

        #myfilename = '/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/cluster_patch/train_log/eval/tp_patch.pickle'
        #(true_positives_B, true_positives_NB) = pickle.load(open(myfilename, 'rb'))

        data_B = vis_square(np.array(true_positives_B[:64]))
        data_NB = vis_square(np.array(true_positives_NB[:64]))
        fig = plt.figure(); plt.imshow(data_B); plt.grid(b=None)
        fig = plt.figure(); plt.imshow(data_NB); plt.grid(b=None)
        plt.show()
        return

    
def plot_visualization_close_centers():

    myfilename = '/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/cluster_patch/train_log/eval/images_closest_to_center_remove_low_contrast_2500.pickle'
    (images_smallest_scores_s, images_largest_scores_s, images_smallest_scores_t, images_largest_scores_t, labels_smallest_scores_s, labels_largest_scores_s, labels_smallest_scores_t, labels_largest_scores_t) = pickle.load(open(myfilename, 'rb'))

    process=0
    smallest=0
    largest=1
    t=0.5
    for k in range(len(images_smallest_scores_s)):
        if smallest:
            data = vis_square(np.array(images_smallest_scores_s[k]).astype(np.uint8),process)
            fig = plt.figure(); plt.imshow(data); plt.grid(b=None)
            print('source:')
            #print(np.array(labels_smallest_scores_s[k]))
            acc_small= sum(np.array(labels_smallest_scores_s[k]).squeeze()>t)/len(labels_smallest_scores_s[k])
            print(acc_small) 


        if largest:
            data = vis_square(np.array(images_largest_scores_s[k]).astype(np.uint8),process)
            fig = plt.figure(); plt.imshow(data); plt.grid(b=None)
            print('source:')
            print(labels_largest_scores_s[k])
            acc_large= sum(np.array(labels_largest_scores_s[k]).squeeze()>t)/len(labels_largest_scores_s[k])
            print(acc_large) 


        if smallest:
            data = vis_square(np.array(images_smallest_scores_t[k]).astype(np.uint8),process)
            fig = plt.figure(); plt.imshow(data); plt.grid(b=None)
            print('target:')
            #print(labels_smallest_scores_t[k])
            acc_small= sum(np.array(labels_smallest_scores_t[k]).squeeze()>t)/len(labels_smallest_scores_t[k])
            print(acc_small)  
           
        if largest:   
            data = vis_square(np.array(images_largest_scores_t[k]).astype(np.uint8),process)
            fig = plt.figure(); plt.imshow(data); plt.grid(b=None)
            print('target:')
            print(labels_largest_scores_t[k])
            acc_large= sum(np.array(labels_largest_scores_t[k]).squeeze()>t)/len(labels_largest_scores_t[k])
            print(acc_large) 
                 
    plt.show()

    


def plot_visualization_close_centers_1():

    myfilename = '/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/configs/cluster_patch/train_log/eval/images_closest_to_center_remove_low_contrast_1500_80_20.pickle'
    (images_smallest_scores_s, images_largest_scores_s, images_smallest_scores_t, images_largest_scores_t, labels_smallest_scores_s, labels_largest_scores_s, labels_smallest_scores_t, labels_largest_scores_t) = pickle.load(open(myfilename, 'rb'))

    process=0
    smallest=1
    largest=1
    t=0.5

    np.random.seed(0)
    
    for k in range(len(images_smallest_scores_s)):
        
        if smallest:
            coords = np.random.choice(len(images_smallest_scores_s[k]), 25, replace=False)
            data = vis_square(np.array([zz for i,zz in enumerate(images_smallest_scores_s[k]) if i in coords]).astype(np.uint8),process)
            fig = plt.figure(); plt.imshow(data); plt.grid(b=None); plt.savefig('smallest_source_{}.png'.format(k)); plt.close()
            print('source:')
            #print(np.array(labels_smallest_scores_s[k]))
            acc_small= sum(np.array(labels_smallest_scores_s[k]).squeeze()>t)/len(labels_smallest_scores_s[k])
            print(acc_small) 

            coords = np.random.choice(len(images_smallest_scores_t[k]), 25, replace=False)
            data = vis_square(np.array([zz for i,zz in enumerate(images_smallest_scores_t[k]) if i in coords]).astype(np.uint8),process)
            fig = plt.figure(); plt.imshow(data); plt.grid(b=None); plt.savefig('smallest_target_{}.png'.format(k)); plt.close()
            print('target:')
            #print(labels_smallest_scores_t[k])
            acc_small= sum(np.array(labels_smallest_scores_t[k]).squeeze()>t)/len(labels_smallest_scores_t[k])
            print(acc_small)  

            

        if largest:
            coords = np.random.choice(len(images_largest_scores_s[k]), 25, replace=False)
            data = vis_square(np.array([zz for i,zz in enumerate(images_largest_scores_s[k]) if i in coords]).astype(np.uint8),process)
            fig = plt.figure(); plt.imshow(data); plt.grid(b=None); plt.savefig('largest_source_{}.png'.format(k)); plt.close()
            print('source:')
            print(labels_largest_scores_s[k])
            acc_large= sum(np.array(labels_largest_scores_s[k]).squeeze()>t)/len(labels_largest_scores_s[k])
            print(acc_large) 

            coords = np.random.choice(len(images_largest_scores_t[k]), 25, replace=False)
            data = vis_square(np.array([zz for i,zz in enumerate(images_largest_scores_t[k]) if i in coords]).astype(np.uint8),process)
            fig = plt.figure(); plt.imshow(data); plt.grid(b=None); plt.savefig('largest_target_{}.png'.format(k)); plt.close()
            print('target:')
            print(labels_largest_scores_t[k])
            acc_large= sum(np.array(labels_largest_scores_t[k]).squeeze()>t)/len(labels_largest_scores_t[k])
            print(acc_large) 
            

        # if smallest:
        #     data = vis_square(np.array(images_smallest_scores_t[k]).astype(np.uint8),process)
        #     fig = plt.figure(); plt.imshow(data); plt.grid(b=None)
        #     print('target:')
        #     #print(labels_smallest_scores_t[k])
        #     acc_small= sum(np.array(labels_smallest_scores_t[k]).squeeze()>t)/len(labels_smallest_scores_t[k])
        #     print(acc_small)  
           
        # if largest:   
        #     data = vis_square(np.array(images_largest_scores_t[k]).astype(np.uint8),process)
        #     fig = plt.figure(); plt.imshow(data); plt.grid(b=None)
        #     print('target:')
        #     print(labels_largest_scores_t[k])
        #     acc_large= sum(np.array(labels_largest_scores_t[k]).squeeze()>t)/len(labels_largest_scores_t[k])
        #     print(acc_large) 
                 
    #plt.show()

    
def adabn_plot(num):
    jet = cm = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=num)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    x = range(0,61,10)
    #y = [56.92, 52.38, 54.86, 55.51, 51.07, 52.52]
    #y =[54.99, 57.02, 58.51, 58.66, 53.91, 48.88]
    #no_adabn_mean = [57.17, 57.58, 58.31]
    #no_adabn = [57.61, 58.33, 58.80]
    #colors = ['r','g','b']
    x_name = 'Adapted Layers'
    y_name='Accuracy'
    title_name='AdaBN upto K layers'
    plt.figure()
    

    for i in range(-1,num):
        modeldir = 'train_log_original_{}'.format(i+1)
        #modeldir = 'train_log_finetune_random'.format(i+1)
        print(modeldir)
        score_path = os.path.join(modeldir,'scores_all_epochs.pickle')
        adabn_path = os.path.join(modeldir,'models','{}_adabn.pickle'.format(modeldir))
        
        if os.path.exists(score_path):
            scores = pickle.load(open(score_path,'rb'))
            adabn = pickle.load(open(adabn_path,'rb'))
            colorVal = scalarMap.to_rgba(i)
            y=[]
            y.append(scores[adabn[1]][1][1])
            y.extend(adabn[0])
            y = np.array(y)*100
            
            #y.append(no_adabn_mean[ii]) #y_all = pickle.load(open('train_log/nst_{}_mean_adabn.pickle'.format(ii+1),'rb')# y.extend(np.array(y_all)*100)
            plt.plot(x, y, '*-', color = colorVal, label='random seed {}'.format(i+1))
            #plt.axhline(60.15,color='r',label='No AdaBN')# red line 
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title_name)

    plt.legend()
    plt.savefig('{}.png'.format(modeldir)) #.format(save_path,metric_value_name,np.max(K)))

    #xlim((left, right)) 
    plt.show()


def epoch_plot(num, stop_val,str0,source_or_test):
    jet = cm = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=num)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    plt.figure()        
    for i in range(0,num):
        
        modeldir = 'train_log_{}{}'.format(str0,i+1) #'train_log_n{}'.format(i+1)
        #modeldir = 'train_log_finetune_random'.format(i+1)
        print(modeldir)
        score_path = os.path.join(modeldir,'scores_all_epochs_{}.pickle'.format(stop_val))

        x=[]; y=[]; y_v=[]
        if os.path.exists(score_path):
            scores = pickle.load(open(score_path,'rb'))
            for k in sorted(list(scores.keys()), key=lambda s: int(s[5:-4])):
                x.append(int(k[5:-4]))
                y.append(scores[k][source_or_test][1]) #'epoch{}.pth'.format(k)])
                y_v.append(scores[k][2]['IoU'])

            colorVal = scalarMap.to_rgba(i)
            plt.plot(x, y, '*-',color = colorVal, label='test {}'.format(i))#label=modeldir)
            plt.plot(x, y_v, '*--',color = colorVal, label='val {}'.format(i))#label=modeldir)
            plt.xlim([400,1250])
    plt.legend()
    plt.savefig('{}_source_val_part_{}.png'.format(str0,stop_val))
    plt.show()



def adabn_plot_detailed(num, str0, str1, str2, start):
    jet = cm = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=start+num+1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    x = range(0,61,1)
    x_name = 'Adapted Layers'
    y_name='Accuracy'
    title_name='AdaBN upto K layers'
    #plt.figure()

    sample_numbers = [100000000] #, 1024, 512, 256, 128, 64]

    for no,sn in enumerate(sample_numbers):
        for i in range(start,start+num):
            modeldir = 'train_log_{}{}'.format(str0,i+1)
            print(modeldir)
            score_path = os.path.join(modeldir,'scores_all_epochs_45.pickle')
            adabn_path = os.path.join(modeldir,'models','{}_{}_adabn{}.pickle'.format(modeldir, sn, str1))
            #pdb.set_trace()
            if os.path.exists(adabn_path):

                scores = pickle.load(open(score_path,'rb'))
                adabn = pickle.load(open(adabn_path,'rb'))
                colorVal = scalarMap.to_rgba(i)
                y=[]
                #pdb.set_trace()
                no_adabn_score = scores[adabn[1]][1][1]
                y.append(no_adabn_score)
                y.extend(adabn[0]-no_adabn_score)
                diff_len  = len(x)-len(y)
                if diff_len>0:
                    #pdb.set_trace()
                    add_arr = np.repeat(y[-1:],diff_len)
                    y.extend(add_arr)
                    
                y = np.array(y)*100
            
                plt.plot(x, y, str2, color = colorVal, label='seed {} '.format(i+1))

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title_name)

    plt.legend()
    #plt.savefig(os.path.join(modeldir,'{}_512.png'.format(modeldir))) #.format(save_path,metric_value_name,np.max(K)))

    #xlim((left, right)) 
    #plt.show()


def adabn_plot_errorbar(num, str0, str1, str2, add):
    jet = cm = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=num)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    x = range(1,61,1)
    x_name = 'Adapted Layers'
    y_name='Accuracy difference'
    title_name='AdaBN up to K'
    #fig, ax = plt.subplots()

    sample_numbers = [100000000] #[2048, 1024, 512, 256, 128, 64]
    
    for no, sn in enumerate(sample_numbers):
        y_all = []
        for i in range(-1,num):
            modeldir = 'train_log_{}{}'.format(str0,i+1)
            print(modeldir)
            score_path = os.path.join(modeldir,'scores_all_epochs_45.pickle')
            adabn_path = os.path.join(modeldir,'models','{}_{}_adabn{}.pickle'.format(modeldir, sn, str1))

            if os.path.exists(adabn_path):
                scores = pickle.load(open(score_path,'rb'))
                adabn = pickle.load(open(adabn_path,'rb'))
                colorVal = scalarMap.to_rgba(i)
                y=[]

                no_adabn_score = scores[adabn[1]][1][1]
                #y.append(no_adabn_score)

                if add==1:
                    y.append(0)
                    y.extend(np.diff(adabn[0]))
                else:
                    y.extend(adabn[0]-no_adabn_score)

                    


                diff_len  = len(x)-len(y)
                if diff_len>0:
                    pdb.set_trace()
                    add_arr = np.repeat(y[-1:],diff_len)
                    y.extend(add_arr)
                    
                y = np.array(y)*100
                y_all.append(y)


        y_all_np = np.array(y_all)
        y_mean = np.mean(y_all_np,axis=0)
        y_err = np.std(y_all_np,axis=0)
        trans = Affine2D().translate(0.2*add, 0) + ax.transData
        #pdb.set_trace()        
        plt.errorbar(x, y_mean, yerr=y_err, label=str2, transform=trans)

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title_name)
    plt.legend()
    plt.savefig(os.path.join(modeldir,'{}_uptok_diff.png'.format(modeldir))) #.format(save_path,metric_value_name,np.max(K)))

    #xlim((left, right)) 


if __name__=='__main__':

    num=20
    str0 = 'original_'
    source_or_test=0
    stop_val=45
    fig, ax = plt.subplots()
    adabn_plot_errorbar(num, str0,'' ,'Up to layer K', 1)
    #adabn_plot_errorbar(num, str0,'_k','At layer K', 2)
    #adabn_plot_errorbar(num, str0,'_k_1','At layers 1 and K', 3)
    
    #adabn_plot_detailed(20, str0,'' ,'*-',-1)
    #adabn_plot_detailed(20, str0,'_k','o-',-1)
    #adabn_plot_detailed(20, str0,'_k_1','*--',-1)
    
    plt.show()

    #adabn_plot_detailed(num, str0)
    
    #epoch_plot(num,stop_val,str0,source_or_test)

    
