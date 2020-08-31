import argparse
import os
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
# from mypath import Path
from common import config
from data import make_data_loader
from model.sync_batchnorm.replicate import patch_replication_callback
from model.deeplab import *
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator
import json
#import visdom
import torch
import pdb
import matplotlib.pyplot as plt

class Trainer(object):
    def __init__(self, config, args):
        self.args = args
        self.config = config
        #self.vis = visdom.Visdom(env=os.getcwd().split('/')[-1])
        # Define Dataloader
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(config)


    def training(self, if_mean):
        train_loss = 0.0
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        #mean_all = [0.29290962, 0.33944466, 0.37961883] v2no transform #[0.32489884, 0.36593837, 0.39595861] transformed_clean
        mean_all = [0.2415185,  0.27953407, 0.31290946]
        
        if not config.transform_sample:
            mean_all = np.array(mean_all)*255
        
        len_image=[]; sum_image=[]; sum_diff=[]
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            for bb in range(image.shape[0]):
                # Note: turn of tr.Normalize() in spacenet.py before running this
                im = image[bb,:].cpu().numpy()
                if if_mean:
                    if config.transform_sample:
                        sum_image.append(np.sum(im,axis=(1,2)))                        
                    else:                            
                        sum_image.append(np.sum(im,axis=(0,1)))
                else:
                    if config.transform_sample:                    
                        mean_diff = im.transpose(1,2,0) - mean_all #
                        #sum_diff.append(np.sum(mean_diff**2, axis=(1,2)))
                    else:
                        mean_diff = im - mean_all #transpose(1,2,0)                        
                    sum_diff.append(np.sum(mean_diff**2, axis=(0,1))) 


                if config.transform_sample:                    
                    len_image.append(len(im[0,:].reshape(-1)))
                else:
                    len_image.append(len(im[:,:,0].reshape(-1)))
                    

        
        len_all = sum(len_image)
        if if_mean:
            assert len(sum_image)==len(len_image)
            sum_all = sum(sum_image)
            mean_all = sum_all/len_all
            print(mean_all/255)
            print(mean_all)
        else:
            assert len(sum_diff)==len(len_image) 
            sum_diff_all = sum(sum_diff)
            mean_all = sum_diff_all/len_all
            mean_all_sqrt = np.sqrt(mean_all)
            print(mean_all_sqrt)
            
        pdb.set_trace()
        return
                                 #self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([train_loss]), win='loss', name='train',
            #          opts=dict(title='loss', xlabel='epoch', ylabel='loss'),
            #          update='append' if epoch > 0 else None)



def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    # training hyper params
    parser.add_argument('--if_mean', type=int, default=0,
                        metavar='N', help='if mean or std calculation (default:1)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
    False, help='disables CUDA training')
    parser.add_argument('--gpu', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        print('Using cuda device:', args.gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if_mean = args.if_mean
    
    print(args)
    trainer = Trainer(config, args)

    trainer.training(if_mean)
        # if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
        #trainer.validation(epoch)


if __name__ == "__main__":
    main()



