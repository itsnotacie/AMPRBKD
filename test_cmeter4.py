import os
import argparse
import time
import math
import copy

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from PIL import Image
from skimage.transform import rotate,resize
from glob import glob
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

#import lib.models as models
#from lib.config import config, update_config
from config.defaults import _C as config
from config.defaults import update_config
from hrnet import get_face_alignment_net


scale = 4
colors = [(0,0,255), (0,255,0), (0,255,255)]

test_transform = transforms.Compose([
    
    transforms.Resize((256,256)),
    transforms.ToTensor()
])
IMG_SIZE_W = IMG_SIZE_H = 256

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description='cmeter')
    parser.add_argument('--cfg', default='HRNet-Facial-Landmark-Detection/experiments/300w/face_alignment_300w_hrnet_w18.yaml', type=str)  
    parser.add_argument('--weights', default='weights/dial_all_0623/dial_all_hr_s_v2_300_0623.pth', type=str)  

    parser.add_argument('--img_input', default='dial/example/', type=str)  
    args = parser.parse_args()
    return args

def test(cfg, model_file, img_input, dist, save_dir,tpye):
    update_config(config, cfg)
    net = get_face_alignment_net(config)

    net.load_state_dict(torch.load(model_file))
    net.eval()
    net.cuda()

    time_start1 = time.time()

    for imgname in glob(img_input + '*.jpg'):
        name = imgname.split('/')[-1]

        img = cv2.imread(imgname)
        h,w,c = img.shape

        img = cv2.resize(img,(256,256))

        data = copy.deepcopy(img)
        data = data.transpose((2,0,1))
        data = torch.FloatTensor(data).unsqueeze(0).cuda()
        pred = net(data)
        hm = pred
       
        hm = hm.cpu().detach().numpy().squeeze(0)
        pt1,pt2,pt3,pt4 = hm[0], hm[1], hm[2], hm[3]
        heatmap = pt1+pt2+pt3+pt4
        save_name = save_dir + name[:-4] + '_heatmap.jpg'
        #print(name,img.shape)
        cv2.imwrite(save_name,heatmap*255)

        idx1 = np.unravel_index(pt1.argmax(), pt1.shape)
        idx2 = np.unravel_index(pt2.argmax(), pt2.shape)
        idx3 = np.unravel_index(pt3.argmax(), pt3.shape) 
        idx4 = np.unravel_index(pt4.argmax(), pt4.shape)


        r0 = (idx1[1] * scale * w / IMG_SIZE_W, idx1[0] * scale * h / IMG_SIZE_H)
        r1 = (idx2[1] * scale * w / IMG_SIZE_W, idx2[0] * scale * h / IMG_SIZE_H)
        r2 = (idx3[1] * scale * w / IMG_SIZE_W, idx3[0] * scale * h / IMG_SIZE_H)
        r3 = (idx4[1] * scale * w / IMG_SIZE_W, idx4[0] * scale * h / IMG_SIZE_H)


        img = cv2.imread(imgname)
        img = cv2.circle(img,(int(r0[0]),int(r0[1])), 5, (255,0,255), -1)
        img = cv2.circle(img,(int(r1[0]),int(r1[1])), 5, (0,255,255), -1)
        img = cv2.circle(img,(int(r2[0]),int(r2[1])), 5, (255,255,0), -1)
        img = cv2.circle(img,(int(r3[0]),int(r3[1])), 5, (0,0,255), -1)

        name = save_dir + name[:-4] + '_pre.jpg'
        #print(name,img.shape)
        cv2.imwrite(name,img)

#怀疑错误
def main():
    args = parse_args()

    model_file = args.weights#'weights/hmeter_230728/hmeter_best_0728.pth'#

    cfg = args.cfg#'HRNet-Facial-Landmark-Detection/experiments/300w/face_alignment_300w_hrnet_w18_hmeter_7.yaml'

    dist = 1
    #filename = args.test_txt #'dial/hr_hmeter_test_1.txt' #val.txt和val文件夹//hr_dial_all_val
    img_input = args.img_input
    save_dir = 'dial/output/'

    test(cfg, model_file,img_input, dist, save_dir, 'test')


if __name__ == '__main__':
    main()