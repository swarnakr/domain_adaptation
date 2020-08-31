import os
import sys
sys.path.append(os.getcwd())
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
from PIL import Image
import pickle
import copy

class Test:
    def __init__(self, model_path, config, bn, save_path, save_batch, sample_number, trial=100, cuda=False):
        self.bn = bn
        self.target=config.all_dataset
        self.target.remove(config.dataset)
        self.sample_number = sample_number
        # load source domain
        #self.source_set = spacenet.Spacenet(city=config.dataset, split='test', img_root=config.img_root, needs to be changed)
        #self.source_loader = DataLoader(self.source_set, batch_size=16, shuffle=False, num_workers=2)
        self.source_loader = None
        self.save_path = save_path
        self.save_batch = save_batch
        self.trial = trial
        self.target_set = []
        self.target_loader = []

        self.target_trainset = []
        self.target_trainloader = []

        self.config = config

        # load other domains
        for city in self.target:
            test = spacenet.Spacenet(city=city, split='val', img_root=config.img_root, gt_root = config.gt_root, mean_std=config.mean_std, if_augment=config.if_augment, repeat_count=config.repeat_count)
            self.target_set.append(test)
            self.target_loader.append(DataLoader(test, batch_size=16, shuffle=False, num_workers=2))

            train = spacenet.Spacenet(city=city, split='train', img_root=config.img_root, gt_root = config.gt_root, mean_std=config.mean_std, if_augment=config.if_augment, repeat_count=config.repeat_count, sample_number= sample_number)
            self.target_trainset.append(train)
            self.target_trainloader.append(DataLoader(train, batch_size=16, shuffle=False, num_workers=2))

            
        self.model = DeepLab(num_classes=2,
                backbone=config.backbone,
                output_stride=config.out_stride,
                sync_bn=config.sync_bn,
                freeze_bn=config.freeze_bn)
        if cuda:
            self.checkpoint = torch.load(model_path)
        else:
            self.checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        #print(self.checkpoint.keys())
        self.model.load_state_dict(self.checkpoint['model'])
        self.evaluator = Evaluator(2)
        self.cuda = cuda
        if cuda:
            self.model = self.model.cuda()


    def get_source_bn(self, trainloader, model, city):
        dirname = os.path.join(self.save_path, city + '_bn')
        bn_before= {}
        model.eval()
        for sample in trainloader:
            image, target, path = sample['image'], sample['label'], sample['path']
            if self.cuda:
                image, target = image.cuda(), target.cuda()
                output = model(image)
                
            layr=0
            for h in model.modules():
                if isinstance(h, nn.Conv2d): 
                    bn_before[(layr),'weight'] = np.squeeze(h.weight.detach().cpu().numpy())
                if isinstance(h, nn.BatchNorm2d): #Conv2d):
                    bn_before[(layr,'mean')] = h.running_mean.cpu().numpy()
                    bn_before[(layr,'var')] = h.running_var.cpu().numpy()
                    layr +=1
            pickle.dump(bn_before,open(os.path.join(dirname, 'bnAll_before_{}.pickle'.format(converged_model)),'wb'));
            break
        return
        
                
    def get_performance(self, dataloader, trainloader, city, adabn_layer,model):
        # change mean and var of bn to adapt to the target domain
        dirname = os.path.join(self.save_path, city + '_bn')
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        model.eval()        
        current_adabn = 0; 
        for h in model.modules():
            if isinstance(h, SynchronizedBatchNorm2d) or isinstance(h, nn.BatchNorm2d):
                current_adabn += 1
                if  current_adabn== adabn_layer: #current_adabn ==1 or
                    h.train()
            
        if self.bn and city != self.config.dataset:
            print('BN Adaptation on' + city)
            for i,sample in enumerate(trainloader):
                image, target, path = sample['image'], sample['label'], sample['path']
                if self.cuda:
                    image, target = image.cuda(), target.cuda()
                with torch.no_grad():
                    output = model(image)

        #bn={}
        #model.eval()
        #for sample in trainloader:
        #    image, target, path = sample['image'], sample['label'], sample['path']
        #    if self.cuda:
        #        image, target = image.cuda(), target.cuda()
        #        output = model(image)
                    
        #    layr=0
        #    for h in model.modules():
        #        if isinstance(h, nn.Conv2d): 
        #            bn[(layr,'weight')] = np.squeeze(h.weight.detach().cpu().numpy())
        #        if isinstance(h, nn.BatchNorm2d): #Conv2d):
        #            bn[(layr,'mean')] = h.running_mean.cpu().numpy()
        #            bn[(layr,'var')] = h.running_var.cpu().numpy()
        #            layr +=1

        #    pickle.dump(bn,open(os.path.join(dirname, 'bnAll_after_k_1_{}_{}_{}.pickle'.format(converged_model.rstrip('.pth'), self.sample_number, adabn_layer)),'wb')); #self.trial
        #    break


        batch = self.save_batch
        model.eval()
        self.evaluator.reset()
        tbar = tqdm(dataloader, desc='\r')

        # save in different directories
        if self.bn:
            save_path = os.path.join(self.save_path, city + '_bn')
        else:
            save_path = os.path.join(self.save_path, city)

       # evaluate on the test dataset
        for i, sample in enumerate(tbar):
            image, target, path = sample['image'], sample['label'], sample['path']
            #print(path)
            if self.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = model(image)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)


            # save pictures
            if batch > 0:
                if not os.path.exists(self.save_path):
                    os.mkdir(self.save_path)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                image = image.cpu().numpy() * 255
                image = image.transpose(0,2,3,1).astype(int)

                imgs = self.color_images(pred, target)
                self.save_images(imgs, batch, save_path, False)
                self.save_images(image, batch, save_path, True)
                batch -= 1

        Acc = self.evaluator.Building_Acc()
        IoU = self.evaluator.Building_IoU()
        mIoU = self.evaluator.Mean_Intersection_over_Union()

        #state = { 'model': model.state_dict()}
        #torch.save(state, os.path.join(dirname, 'weights_{}_{}_after_adabn_k_{}.pth'.format(converged_model.rstrip('.pth'), self.sample_number, adabn_layer))) 

        return Acc, IoU, mIoU

    def test(self, name, converged_model):
        main_model = self.model

        source_test=0
        if source_test:
            A, I, Im = self.get_performance(self.source_loader, None, self.config.dataset)

        iter_model = copy.deepcopy(main_model)
        self.get_source_bn(self.target_trainloader[0], iter_model, self.target[0])
        
        i_all=[]
        for l in range(1,113,1): #in range(1,61,1):#(10,61,10):
            tA, tI, tIm = [], [], []

            for dl, tl, city in zip(self.target_loader, self.target_trainloader, self.target):
                iter_model = copy.deepcopy(main_model)
                tA_, tI_, tIm_ = self.get_performance(dl, tl, city, l, iter_model)

                tA.append(tA_)
                tI.append(tI_)
                tIm.append(tIm_)
            i_all.append(tI[0])

            name_f = '{}/{}_adabn_k.pickle'.format(name,city)
            pickle.dump((i_all,converged_model,self.sample_number), open(name_f,'wb'))
        #pickle.dump((i_all,converged_model,self.sample_number), open(name,'wb'))
        
        res = {}
        if source_test:
            print("Test for source domain:")
            print("{}: Acc:{}, IoU:{}, mIoU:{}".format(self.config.dataset, A, I, Im))
            res[config.dataset] = {'Acc': A, 'IoU': I, 'mIoU':Im}

        print('Test for target domain:')
        for i, city in enumerate(self.target):
            print("{}: Acc:{}, IoU:{}, mIoU:{}".format(city, tA[i], tI[i], tIm[i]))
            res[city] = {'Acc': tA[i], 'IoU': tI[i], 'mIoU': tIm[i]}

        if self.bn:
            name = 'train_log/test_bn.json'
        else:
            name = 'train_log/test.json'

        with open(name, 'w') as f:
            json.dump(res, f)

    def save_images(self, imgs, batch_index, save_path, if_original=False):
        for i, img in enumerate(imgs):
            #img = img[:,:,::-1] # change to BGR
            #from IPython import embed
            #embed()
            if not if_original:
                cv2.imwrite(os.path.join(save_path, str(batch_index) + str(i) + '_Original.jpg'), img)
            else:
                img = img.astype(np.uint8)
                img = Image.fromarray(img)
                img.save(os.path.join(save_path, str(batch_index) + str(i) + '_Pred.jpg'))
                #cv2.imwrite(os.path.join(save_path, str(batch_index) + str(i) + '_Pred.jpg'), img)

    def color_images(self, pred, target):
        imgs = []
        for p, t in zip(pred, target):
            tmp = p * 2 + t
            np.squeeze(tmp)
            img = np.zeros((p.shape[0], p.shape[1], 3))
            # bkg:negative, building:postive
            #from IPython import embed
            #embed()
            img[np.where(tmp==0)] = [0, 0, 0] # Black RGB, for true negative
            img[np.where(tmp==1)] = [255, 0, 0] # Red RGB, for false negative
            img[np.where(tmp==2)] = [0, 255, 0] # Green RGB, for false positive
            img[np.where(tmp==3)] = [255, 255, 0] #Yellow RGB, for true positive
            imgs.append(img)
        return imgs

if __name__ == "__main__":
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument('modeldir', default='train_log')
    parser.add_argument('--sample_number', default='1000000', type=int)
    #parser.add_argument('trial',type=int)
    
    #parser.add_argument('bn', default=True, type=str2bool,
            #help='whether to use BN adaptation')
    #parser.add_argument('save_batch', default=0, type=int,
            #help='number of test images to save (n*batch_size)')
    
    parser.add_argument('--cuda', default=True,
            help='whether to use GPU')
    #parser.add_argument('--save_path', default='train_log/test_images/',
    #        help='path to save images')
    
    args = parser.parse_args()
    stop_val = 45
    
    bn = True
    save_batch = 0
    save_path = os.path.join(args.modeldir, 'test_images')
    all_results = os.listdir(os.path.join(args.modeldir, 'models'))

    all_results = [file_1 for file_1 in all_results if not file_1.endswith(".pickle")]

    all_results = sorted(all_results, key=lambda x: int(x[5:-4]), reverse=True)
    epoch_numbers = [int(x[5:-4]) for x in all_results]
    epoch_differences = [epoch_numbers[i] - epoch_numbers[i+1] for i in range(len(epoch_numbers)-1)] + [0]
    my_stop_point=0
    #for jj,r,e,d in zip(range(len(all_results)), all_results, epoch_numbers, epoch_differences):
    #    if d>=stop_val:
    #        my_stop_point = jj+1
            
    #converged_model = all_results[my_stop_point+1]
    #stop_epoch_number = epoch_numbers[my_stop_point]

    converged_model = all_results[0]
    #num = np.minimum(20,len(all_results))
    
    fullpath = os.path.join(args.modeldir, 'models', converged_model)
    test = Test(fullpath, config, bn, save_path, save_batch, args.sample_number, cuda=args.cuda)
    name_path = os.path.join(args.modeldir, 'models')

    if 1: #not os.path.exists(name_path):
        test.test(name_path, converged_model)

