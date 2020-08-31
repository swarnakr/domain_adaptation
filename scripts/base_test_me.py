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
import matplotlib.pyplot as plt
import numpy.matlib as mt

activation=[];
def save_output2(module, input, output):
     global activation
     #pdb.set_trace();
     activation = output
     return 

class Test:
    def __init__(self, model_path, config, bn, save_path, save_batch, cuda=False):
        self.bn = bn
        self.target=config.all_dataset
        self.target.remove(config.dataset)
        # load source domain
        self.source_set = spacenet.Spacenet(city=config.dataset, split='test', img_root=config.img_root)
        self.source_loader = DataLoader(self.source_set, batch_size=16, shuffle=False, num_workers=2)

        self.save_path = save_path
        self.save_batch = save_batch

        self.target_set = []
        self.target_loader = []

        self.target_trainset = []
        self.target_trainloader = []

        self.config = config

        # load other domains
        for city in self.target:
            test = spacenet.Spacenet(city=city, split='test', img_root=config.img_root)
            self.target_set.append(test)
            self.target_loader.append(DataLoader(test, batch_size=16, shuffle=False, num_workers=2))
            train = spacenet.Spacenet(city=city, split='train', img_root=config.img_root)
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
        
        self.model.load_state_dict(self.checkpoint)
        self.evaluator = Evaluator(2)
        self.cuda = cuda
        if cuda:
            self.model = self.model.cuda()

    def save_output(module, input, output):
     global activation,i
     # save output
     print('I came here')
     pdb.set_trace();
     channels = output.permute(1, 0, 2, 3)
     c = channels.shape[0]
     features = channels.reshape(c, -1)
     if len(activation) == i:
        activation.append(features)
     else:
        activation[i] = torch.cat([activation[i], features], dim=1)
     i += 1
     return 

    def get_performance(self, dataloader, trainloader, city):
        # change mean and var of bn to adapt to the target domain
        #pdb.set_trace()
        if self.bn:
            save_path = os.path.join(self.save_path, city + '_bn')
        else:
            save_path = os.path.join(self.save_path, city)

        if self.bn and city != self.config.dataset: #!= self.config.dataset:
            print('BN Adaptation on' + city)
            self.model.train()
            layr=0
            for h in self.model.modules():
                if isinstance(h, nn.ReLU6): #Conv2d):
                   layr+=1
                   if layr == 1:
                      h.register_forward_hook(save_output2)
                   if layr > 1:
                      break;
            tbar = tqdm(dataloader, desc='\r')

            for sample in trainloader:
#            for i, sample in enumerate(tbar):
                image, target = sample['image'], sample['label']
                pdb.set_trace()
     #           add0 = np.tile([10, 42, 37],(400,400,16,1));
      #          add = add0.transpose(2,3,0,1)
       #         image = image + add;
                if self.cuda:
                    image, target = image.cuda(), target.cuda()
                with torch.no_grad():
                    output = self.model(image)
                if not os.path.exists(save_path):
                   os.mkdir(save_path)

                pdb.set_trace()	
                self.save_act(activation, save_path, False)
                self.save_act(image.numpy() * 255, save_path, True)



        batch = self.save_batch
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(dataloader, desc='\r')

        # save in different directories
        if self.bn:
            save_path = os.path.join(self.save_path, city + '_bn')
        else:
            save_path = os.path.join(self.save_path, city)

        layr=0
        for h in self.model.modules():
            if isinstance(h, nn.ReLU6): #Conv2d):
               layr+=1
               if layr == 2:
                  h.register_forward_hook(save_output2)
               if layr > 2:
                  break;

       # evaluate on the test dataset
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            pdb.set_trace()
            if not os.path.exists(save_path):
               os.mkdir(save_path)

            self.save_act(activation, save_path, False)
            self.save_act(image.numpy() * 255, save_path, True)

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
        return Acc, IoU, mIoU

    def test(self):
        #A, I, Im = self.get_performance(self.source_loader, None, self.config.dataset)
        tA, tI, tIm = [], [], []
        for dl, tl, city in zip(self.target_loader, self.target_trainloader, self.target):
            tA_, tI_, tIm_ = self.get_performance(dl, tl, city)
            tA.append(tA_)
            tI.append(tI_)
            tIm.append(tIm_)

        res = {}
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


    def save_act(self, imgs, save_path,ifImage):
        if ifImage:
          for i, img in enumerate(imgs):
            img = img.transpose(1,2,0)
            img = img[:,:,::-1] 
            cv2.imwrite(os.path.join(save_path, 'im'+str(i) + '.jpg'), img) 

        else:
          for i, img in enumerate(imgs):
            for j,act in enumerate(img): 
               cv2.imwrite(os.path.join(save_path, 'im'+str(i) + 'act'+ str(j) + '.jpg'), act.numpy()*255)

    def save_images(self, imgs, batch_index, save_path, if_original=False):
        for i, img in enumerate(imgs):
            img = img[:,:,::-1] # change to BGR
            #from IPython import embed
            #embed()
            if not if_original:
                cv2.imwrite(os.path.join(save_path, str(batch_index) + str(i) + '_Original.jpg'), img)
            else:
                cv2.imwrite(os.path.join(save_path, str(batch_index) + str(i) + '_Pred.jpg'), img)

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
    parser.add_argument('model', default='train_log/models/epoch1.pth')
    parser.add_argument('bn', default=True, type=str2bool,
            help='whether to use BN adaptation')
    parser.add_argument('save_batch', default=0, type=int,
            help='number of test images to save (n*batch_size)')
    parser.add_argument('--cuda', default=False,
            help='whether to use GPU')
    parser.add_argument('--save_path', default='./train_log/test_images/',
            help='path to save images')
    args = parser.parse_args()
    test = Test(args.model, config, args.bn, args.save_path, args.save_batch, args.cuda)
    test.test()
    

