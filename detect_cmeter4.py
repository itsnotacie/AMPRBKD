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
from dataset_v2 import HMDataset

import random
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description='hmeter')
    parser.add_argument('--cfg', default='HRNet-Facial-Landmark-Detection/experiments/300w/face_alignment_300w_hrnet_w18.yaml', type=str)  
    parser.add_argument('--weights', default='', type=str)

    parser.add_argument('--type', default='test', type=str)  

    parser.add_argument('--dist', type=float, default=1, help='dist')

    parser.add_argument('--gpus', type=str, default='1', help='gpu id')

    parser.add_argument('--path', default='dial/', type=str)

    parser.add_argument('--val_txt', default='dial/hr_cmeter_test.txt', type=str)

    parser.add_argument('--out_path', default='dial/output/', type=str)

    parser.add_argument('--size', type=int, default=256, help='size')

    args = parser.parse_args()
    return args

def ap(dis1,dis2,dis3,dis4,dis,TP,FP,TP_T,FP_T):
    flag = True
    if dis1 < dis: TP = TP+1
    else:
        FP = FP+1
        flag = False

    if dis2 < dis: TP = TP+1
    else:
        FP = FP+1
        flag = False

    if dis3 < dis: TP = TP+1
    else:
        FP = FP+1
        flag = False

    if dis4 < dis: TP = TP+1
    else:
        FP = FP+1
        flag = False
    
    #break
    if flag: TP_T = TP_T + 1
    else: FP_T = FP_T + 1

    return TP,FP,TP_T,FP_T

def val(args, device, net, filename, dist, imgname_output,tpye):
    time_start1 = time.time()
    scale = args.size / 64



    f = open(filename , "r")
    lines = f.readlines()
    lines = [line.strip().split() for line in lines]

    TP1,FP1,TP_T1,FP_T1 = 0,0,0,0
    TP3,FP3,TP_T3,FP_T3 = 0,0,0,0
    TP5,FP5,TP_T5,FP_T5 = 0,0,0,0
    TP8,FP8,TP_T8,FP_T8 = 0,0,0,0
    TP10,FP10,TP_T10,FP_T10 = 0,0,0,0
    TP15,FP15,TP_T15,FP_T15 = 0,0,0,0
    for ls in lines:
        imgname = filename.split('/')[0] + '/' + ls[0]
        if not os.path.isfile(imgname) : continue
        points=np.array(ls[1:], dtype=float)

        img = cv2.imread(imgname)
        if img is None : continue
        #h,w,c = img.shape

        
        img = cv2.resize(img,(256,256))
        #print('val',img.shape,img[0][0])

        #data = copy.deepcopy(img)
        #data = img.astype(np.uint8)
        data = img.transpose((2,0,1))
        data = torch.FloatTensor(data).unsqueeze(0)#.cuda()

        data = data.to(device)
        pred = net(data)
        hm = pred

        #print('val',data.shape,data[0][0][0])
       
        hm = hm.cpu().detach().numpy().squeeze(0)
        pt1,pt2,pt3,pt4 = hm[0], hm[1], hm[2], hm[3]

        idx1 = np.unravel_index(pt1.argmax(), pt1.shape)
        idx2 = np.unravel_index(pt2.argmax(), pt2.shape)
        idx3 = np.unravel_index(pt3.argmax(), pt3.shape) 
        idx4 = np.unravel_index(pt4.argmax(), pt4.shape)

        #r0 = (idx1[1] * scale * w / IMG_SIZE_W, idx1[0] * scale * h / IMG_SIZE_H)
        r0 = (idx1[1] * scale, idx1[0] * scale)
        r1 = (idx2[1] * scale, idx2[0] * scale)
        r2 = (idx3[1] * scale, idx3[0] * scale)
        r3 = (idx4[1] * scale, idx4[0] * scale)

        #print('val',data.shape,data[0][0][0])
        #print('out : ', r0,r1,r2,r3)
        #print('tag : ', points*256)
        

        dis1 = pow(pow(r0[0] - points[0] *args.size,2) + pow(r0[1] - points[1] *args.size, 2), 0.5)
        dis2 = pow(pow(r1[0] - points[2] *args.size,2) + pow(r1[1] - points[3] *args.size, 2), 0.5)
        dis3 = pow(pow(r2[0] - points[4] *args.size,2) + pow(r2[1] - points[5] *args.size, 2), 0.5)
        dis4 = pow(pow(r3[0] - points[6] *args.size,2) + pow(r3[1] - points[7] *args.size, 2), 0.5)

        #print(dis1,dis2,dis3,dis4)
        if tpye == 'test' and (dis1>10*dist or dis2>10*dist or dis3>10*dist or dis4>10*dist):
            #img = cv2.imread(imgname)
            #img = cv2.resize(img,(64,64))
            line = 4
            img = cv2.circle(img,(int(r0[0]),int(r0[1])), line, (255,0,255), -1)
            img = cv2.circle(img,(int(r1[0]),int(r1[1])), line, (0,255,255), -1)
            img = cv2.circle(img,(int(r2[0]),int(r2[1])), line, (255,255,0), -1)
            img = cv2.circle(img,(int(r3[0]),int(r3[1])), line, (255,255,255), -1)

            '''img = cv2.circle(img,(int(points[0] *IMG_SIZE_W),int(points[1] *IMG_SIZE_H)), line, (200,0,200), -1)
            img = cv2.circle(img,(int(points[2] *IMG_SIZE_W),int(points[3] *IMG_SIZE_H)), line, (0,200,200), -1)
            img = cv2.circle(img,(int(points[4] *IMG_SIZE_W),int(points[5] *IMG_SIZE_H)), line, (200,200,0), -1)
            img = cv2.circle(img,(int(points[6] *IMG_SIZE_W),int(points[7] *IMG_SIZE_H)), line, (200,200,200), -1)'''

            name = imgname_output + ls[0].split('/')[-1]
            cv2.imwrite(name,img)
                
                
        TP1,FP1,TP_T1,FP_T1 = ap(dis1,dis2,dis3,dis4,1*dist,TP1,FP1,TP_T1,FP_T1)
        TP3,FP3,TP_T3,FP_T3 = ap(dis1,dis2,dis3,dis4,3*dist,TP3,FP3,TP_T3,FP_T3)
        TP5,FP5,TP_T5,FP_T5 = ap(dis1,dis2,dis3,dis4,5*dist,TP5,FP5,TP_T5,FP_T5)
        TP8,FP8,TP_T8,FP_T8 = ap(dis1,dis2,dis3,dis4,8*dist,TP8,FP8,TP_T8,FP_T8)
        TP10,FP10,TP_T10,FP_T10 = ap(dis1,dis2,dis3,dis4,10*dist,TP10,FP10,TP_T10,FP_T10)
        TP15,FP15,TP_T15,FP_T15 = ap(dis1,dis2,dis3,dis4,15*dist,TP15,FP15,TP_T15,FP_T15)
        #break

    time_end1=time.time()
    time_end2=time.time()

    if tpye == 'test':
        print("1 像素  -   ",round(TP_T1/(TP_T1+FP_T1), 5),"     :     ",round(TP1/(TP1+FP1), 5), TP_T1,FP_T1,TP1,FP1)
        print("3 像素  -   ",round(TP_T3/(TP_T3+FP_T3), 5),"     :     ",round(TP3/(TP3+FP3), 5), TP_T3,FP_T3,TP3,FP3)
        print("5 像素  -   ",round(TP_T5/(TP_T5+FP_T5), 5),"     :     ",round(TP5/(TP5+FP5), 5), TP_T5,FP_T5,TP5,FP5)
        print("8 像素  -   ",round(TP_T8/(TP_T8+FP_T8), 5),"     :     ",round(TP8/(TP8+FP8), 5), TP_T8,FP_T8,TP8,FP8)
        print("10像素  -   ",round(TP_T10/(TP_T10+FP_T10), 5),"     :     ",round(TP10/(TP10+FP10), 5), TP_T10,FP_T10,TP10,FP10)
        print("15像素  -   ",round(TP_T15/(TP_T15+FP_T15), 5),"     :     ",round(TP15/(TP15+FP15), 5), TP_T15,FP_T15,TP15,FP15)
    if tpye == 'train':
        print("10像素  -   ",round(TP_T10/(TP_T10+FP_T10), 5),"     :     ",round(TP10/(TP10+FP10), 5), TP_T10,FP_T10,TP10,FP10)
    #print('\n')
    #print(len(lines),"  time1 : ",time_end1 - time_start1,'s ',(time_end1 - time_start1)/len(lines))
    #print(len(lines),"  time2 : ",time_end2 - time_start2,'s ',(time_end2 - time_start2)/len(lines))
    f.close()
    return TP_T10/(TP_T10+FP_T10)

def eval_acc(args, out, tag, points):
    scale = int(args.size / 64)
    points = points.cpu().detach().numpy()

    tp = 0
    for i in range(0,len(out)):
        output = out[i].detach().cpu().numpy()
        #target = tag[i].detach().cpu().numpy()
        target = points[i]

        o_pt1,o_pt2,o_pt3,o_pt4 = output[0], output[1], output[2], output[3]
        #t_pt1,t_pt2,t_pt3,t_pt4 = target[0], target[1], target[2], target[3]
        #print(o_pt1.shape,t_pt1.shape)

        o_idx1 = np.unravel_index(o_pt1.argmax(), o_pt1.shape)
        o_idx2 = np.unravel_index(o_pt2.argmax(), o_pt2.shape)
        o_idx3 = np.unravel_index(o_pt3.argmax(), o_pt3.shape) 
        o_idx4 = np.unravel_index(o_pt4.argmax(), o_pt4.shape)

        '''t_idx1 = np.unravel_index(t_pt1.argmax(), t_pt1.shape)
        t_idx2 = np.unravel_index(t_pt2.argmax(), t_pt2.shape)
        t_idx3 = np.unravel_index(t_pt3.argmax(), t_pt3.shape) 
        t_idx4 = np.unravel_index(t_pt4.argmax(), t_pt4.shape)'''

        #print('on val')
        #print('out : ',o_idx1[0]*scale,o_idx1[1]*scale,o_idx2[0]*scale,o_idx2[1]*scale,o_idx3[0]*scale,o_idx3[1]*scale,o_idx4[0]*scale,o_idx4[1]*scale)
        #print('tag : ',t_idx1[0]*scale,t_idx1[1]*scale,t_idx2[0]*scale,t_idx2[1]*scale,t_idx3[0]*scale,t_idx3[1]*scale,t_idx4[0]*scale,t_idx4[1]*scale)
        #print('tag : ',target*256)


        r0 = (o_idx1[1] * scale, o_idx1[0] * scale)
        r1 = (o_idx2[1] * scale, o_idx2[0] * scale)
        r2 = (o_idx3[1] * scale, o_idx3[0] * scale)
        r3 = (o_idx4[1] * scale, o_idx4[0] * scale)

        #print('out : ', r0,r1,r2,r3)
        
        dis1 = pow(pow(r0[0] - target[0] *args.size,2) + pow(r0[1] - target[1] *args.size, 2), 0.5)
        dis2 = pow(pow(r1[0] - target[2] *args.size,2) + pow(r1[1] - target[3] *args.size, 2), 0.5)
        dis3 = pow(pow(r2[0] - target[4] *args.size,2) + pow(r2[1] - target[5] *args.size, 2), 0.5)
        dis4 = pow(pow(r3[0] - target[6] *args.size,2) + pow(r3[1] - target[7] *args.size, 2), 0.5)
        #print(dis1,dis2,dis3,dis4)

        if dis1<2.5*scale and dis2<2.5*scale and dis3<2.5*scale and dis4<2.5*scale :
            tp += 1
    return tp

def onVal_test(args, device,net):
    val_data = HMDataset(args.val_txt, args.path, 'val')
    val_loader = DataLoader(val_data, batch_size=32,shuffle=True, num_workers=8, pin_memory=True, drop_last=False)#batch_size=32,num_workers=8

    correct = 0
    val_acc = 0
    with torch.no_grad():
        for img, tag, points in val_loader:
            img,tag = img.to(device), tag.to(device)
            out = net(img)

            #print(img.shape, img[0][0][0])
            correct += eval_acc(args, out, tag, points)
        val_acc =  correct / len(val_loader.dataset)
    print("10像素  -   ",val_acc,"     :     ",correct, len(val_loader.dataset)-correct)
    return val_acc

#怀疑错误
def main():
    seed_everything(0)
    args = parse_args()

    device = torch.device("cuda:%s" % (args.gpus) if torch.cuda.is_available() else "cpu")

    update_config(config, args.cfg)
    net = get_face_alignment_net(config)

    load_ckpt = torch.load(args.weights)
    net.load_state_dict(load_ckpt)
    net.eval()
    net = net.to(device)

    #model_file = args.weights#'weights/hmeter_230728/hmeter_best_0728.pth'#

    #filename = args.val_txt #'dial/hr_hmeter_test_1.txt' #val.txt和val文件夹//hr_dial_all_val

    val(args, device, net, args.val_txt, args.dist, args.out_path, args.type)

    onVal_test(args, device, net)


if __name__ == '__main__':
    main()