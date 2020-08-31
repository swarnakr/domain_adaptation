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
#import matplotlib.pyplot as plt
import numpy.matlib as mt
import matplotlib.pyplot as plt

activation={};ii=0
def save_output2(module, input, output):
     global activation,ii, randCh, ncity #, first
#     if first==1:
#         randCh[ii] = np.random.randint(output.shape[1])
#     print(randCh[ii])
#     if ncity=='Vegas':
#         pdb.set_trace()
     try:
         feat = output[:,randCh[ii],:]
     except:
         pdb.set_trace()
     activation[ii] = feat
     ii+=1
     return 

class Test:
    def __init__(self, model_path, config, bn, save_path, save_batch, cuda=False):
        self.bn = bn
        self.target=config.all_dataset
        self.target.remove(config.dataset)
        # load source domain
        self.source_set = spacenet.Spacenet(city=config.dataset, split='val', img_root=config.img_root)
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
            test = spacenet.Spacenet(city=city, split='val', img_root=config.img_root)
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
     channels = output.permute(1, 0, 2, 3)
     c = channels.shape[0]
     features = channels.reshape(c, -1)
     if len(activation) == i:
        activation.append(features)
     else:
        activation[i] = torch.cat([activation[i], features], dim=1)
     i += 1
     return 

    def get_performance(self, dataloader, trainloader,city):

        if 1:
            pix = torch.load("pix" +self.config.dataset+ "_" +city + ".pt")
            for k in range(0,len(pix)):
              fig = plt.figure();  plt.hist(pix[k]); plt.xlabel('Activation values'); plt.ylabel("Count"); #plt.legend()
              fig.savefig('./train_log/figs/pix_' + self.config.dataset + '_' + city+ 'act' + str(k) +  '.png')
            return [],[],[]


        # change mean and var of bn to adapt to the target domain
        if self.bn and city != self.config.dataset:
            self.checkpoint = torch.load('./train_log/' + self.config.dataset + '_da_' + city + '.pth')
            self.model.load_state_dict(self.checkpoint)
            if self.cuda:
                self.model = self.model.cuda()

        if 0: #self.bn and city != self.config.dataset:
            print('BN Adaptation on' + city)
            self.model.train()
            tbar = tqdm(dataloader, desc='\r')
            for sample in trainloader:
                image, target = sample['image'], sample['label']
                if self.cuda:
                    image, target = image.cuda(), target.cuda()
                with torch.no_grad():
                    output = self.model(image)

        batch = self.save_batch
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(dataloader, desc='\r')
        # save in different directories
        if self.bn:
            save_path = os.path.join(self.save_path, city + '_bn')
        else:
            save_path = os.path.join(self.save_path, city)

        global randCh; #randCh={};
        global first; first = 1
        global ii; ii=0
        global ncity; ncity = city
        randCh = torch.load('randCh.pt')
        #if city != self.config.dataset:
        #randCh = torch.load('randCh.pt')
        #else: 
         #   randCh={};
        layr=0;  aa=0;
        if city == self.config.dataset:
            for hh in self.model.modules():
                  if isinstance(hh, nn.ReLU6): #Conv2d):
                     layr+=1
                     if layr%5==0:
                         #if first==1 and city == self.config.dataset:
                             #randCh[aa] =  np.random.randint(hh.out_channels)
                         #print(layr)
                         hh.register_forward_hook(save_output2)
                         aa+=1
 
       # evaluate on the test dataset
        pix={} ; pix1 = {};   
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)

            #manipulate activations here
                for k in activation.keys():
                   if first==1:
                      pix[k]=[]; pix1[k]=[];
                   for row in range(0,activation[k].shape[1]):
                       actkrow = activation[k][:,row,:-1].reshape(-1).cpu().numpy()
                       pix[k] = np.hstack((pix[k],actkrow)) 
                       actkrow1 = activation[k][:,row,1:].reshape(-1).cpu().numpy()
                       pix1[k] = np.hstack((pix1[k],actkrow1))
                   for bb in range(0,activation[k].size(0)):
                       cv2.imwrite(os.path.join(save_path, 'act' + str(k) + 'im'+str(i) + 'b'+str(bb) + city + '.jpg'), activation[k][bb,:].cpu().numpy()*255)

                first+=1
                ii=0;
          
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

        corrVal={}
        for k in activation.keys():
            corrVal[k] = np.corrcoef(pix[k], pix1[k])[0,1]
            #_ = plt.hist(pix[k]); plt.show()
        torch.save(pix,"pix" +self.config.dataset+ "_" +city + ".pt")
        print(corrVal)

        Acc = self.evaluator.Building_Acc()
        IoU = self.evaluator.Building_IoU()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        return Acc, IoU, mIoU

    def test(self):
        A, I, Im = self.get_performance(self.source_loader, None, self.config.dataset)
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
    


