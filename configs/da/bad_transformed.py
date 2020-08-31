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
import cv2
from PIL import Image

class Trainer(object):
    def __init__(self, config, args):
        self.args = args
        self.config = config
        #self.vis = visdom.Visdom(env=os.getcwd().split('/')[-1])
        # Define Dataloader
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(config)


    def training(self, if_mean):
        tbar = tqdm(self.train_loader)
        #num_img_tr = len(self.train_loader)

        image_reference = np.random.random((512,512,3))
        image_reference = (image_reference*255).astype(np.uint8)

        hist_reference = cv2.calcHist([image_reference], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_reference = cv2.normalize(hist_reference, hist_reference).flatten()
        ref_value = cv2.compareHist(hist_reference, hist_reference, cv2.HISTCMP_INTERSECT)

        count=0; count_all=0; count_zeros=0;
        len_image=[]; sum_image=[]; sum_diff=[]
        for i, sample in enumerate(tbar):
            image, filenames = sample['image'], sample['path']


 
            for bb in range(image.shape[0]):
                fname = filenames[bb]
                count_all += 1
                if 1:
                    im = image[bb,:].cpu().numpy()
                    total_size = np.prod(im.shape)
                    non_zero_count = np.count_nonzero(im)
                    non_zero_perc = non_zero_count / total_size

                    if 1:
                        pdb.set_trace()
                        im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
                        value_channel = im[:,:,2]
                        hist = cv2.calcHist([value_channel], [0], None, [256], [0, 256])
                        hist_std = np.std(value_channel.reshape(-1))
                        print(hist_std)
                        plt.plot(hist); plt.show()
                        return
                    
                    #hist = cv2.calcHist([im], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                    hist = cv2.normalize(hist, hist).flatten()
                    d = cv2.compareHist(hist, hist_reference, cv2.HISTCMP_INTERSECT)
                    d_ratio = d/ref_value
                    #print('{},{},{}\n'.format(d, d_ratio, fname))
                    if non_zero_perc < 0.5:
                        ind = fname.find('_')
                        #pdb.set_trace()
                        path_transformed = [os.path.join(config.img_root, '{}_out_{}.png'.format(fname[:ind],pp+1)) for pp in range(5)]
                        for path in path_transformed:
                            if os.path.exists(path):                        
                                count_zeros += 1
                                
                    if 0: #d_ratio > 0.05 or non_zero_perc < 0.5:
                        count+=1
                        path_remove = os.path.join(config.img_root, fname)
                        os.remove(path_remove)

                        #im_save = Image.fromarray(im)
                        #im_save = im_save.resize((128,128))
                        #im_save.save(os.path.join('./', 'train_log', 'test_images', 'bad_transformed1', fname))

                    if 0:
                        im_pixels = im.reshape(-1)
                        plt.figure(); plt.imshow(im)
                        plt.figure(); plt.hist(im_pixels) #5000, (1,1000)
                        plt.show()
        print('number bad: {}, and number zeros: {} out of {} images, {}%'.format(count, count_zeros, count_all, count/count_all * 100))

        return



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



