from hrnet import get_face_alignment_net
from config.defaults import _C as config
from config.defaults import update_config
from dataset_v2 import HMDataset
#from dataset_v3 import HMDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import os
import time
import datetime

from detect_cmeter4 import val, onVal_test

import matplotlib.pyplot as plt
import sys
import argparse

import random
import numpy as np


#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def seed_everything(seed = 0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # 禁用benchmark，保证可复现
        # torch.backends.cudnn.benchmark = True # 恢复benchmark，提升效果
        torch.backends.cudnn.enabled = True
    # torch.set_deterministic(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


import warnings
warnings.filterwarnings("ignore")

if hasattr(torch.cuda, 'empty_cache'):
	torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser(description='cmeter')
    parser.add_argument('--cfg', default='HRNet-Facial-Landmark-Detection/experiments/300w/face_alignment_300w_hrnet_w18.yaml', type=str)  

    parser.add_argument('--weights', default='', type=str)  #weights/dial_all_0623/dial_all_hr_s_v2_300_0623.pth
    parser.add_argument('--early_stop', type=int, default=50, help='total training epochs')

    parser.add_argument('--size', type=int, default=256, help='size')

    parser.add_argument('--gpus', type=str, default='1', help='gpu id')

    parser.add_argument('--epochs', type=int, default=100, help='epochs')

    parser.add_argument('--save_path', default='weights/', type=str)
    parser.add_argument('--name', default='cmeter_23test', type=str)

    parser.add_argument('--path', default='dial/', type=str)

    parser.add_argument('--train_txt', default='dial/hr_cmeter.txt', type=str)

    parser.add_argument('--val_txt', default='dial/hr_cmeter_test.txt', type=str)

    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')

    parser.add_argument('--flag_show', nargs='?', const=True, default=True, help='show train heatmap')
    #parser.add_argument('--resume', default=output_path, type=str)  
    args = parser.parse_args()
    return args

def delete_files(folder_path, file_name):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file_name in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)

class HRCmeter():
    def __init__(self, args):
        self.args = args
        # print(config.TRAIN.LR)
        update_config(config, self.args.cfg)
        self.net = get_face_alignment_net(config)
        self.criterion = nn.L1Loss().cuda()
        # self.criterion = nn.MSELoss().cuda()
        # self.criterion = nn.SmoothL1Loss().cuda()
        self.criterion2 = nn.CrossEntropyLoss().cuda()
        self.device = torch.device("cuda:%s" % (self.args.gpus) if torch.cuda.is_available() else "cpu")

        model_dict = self.net.state_dict()
        self.net.load_state_dict(model_dict)
        self.net = self.net.to(self.device)
        # print(config.TRAIN.LR)
        # bb 1264
        self.optimizer = optim.Adam(self.net.parameters(), lr =config.TRAIN.LR)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,step_size=10, gamma=0.9)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
        #                                                      T_0=80, 
        #                                                     T_mult=1)
        #self.optimizer = optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9)
        #self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

    def train(self):
            self.early_stop_value = -1
            self.early_stop_dist = 0
            #self.last_save_path = None
            self.best_epoch = 0
            self.earlystop = False

            train_data = HMDataset(args.train_txt, args.path,'train')
            train_loader = DataLoader(train_data, batch_size=32,shuffle=True, num_workers=8, pin_memory=True, drop_last=False)#batch_size=32,num_workers=8

            val_data = HMDataset(args.val_txt, args.path,'val')
            val_loader = DataLoader(val_data, batch_size=32,shuffle=True, num_workers=8, pin_memory=True, drop_last=False)#batch_size=32,num_workers=8

            for epoch in range(self.args.epochs):
                self.onTrain(train_loader, epoch)
                self.onVal(val_loader, epoch)

                if self.earlystop or epoch == self.args.epochs-1:
                    #val(self.args, self.device, self.net, self.args.val_txt, 1, None, 'test')
                    break

    def onTrain(self, train_loader, epoch):
        self.net.train()

        count = 0
        tp_count = 0
        batch_time = 0
        total_loss = 0

        batch_idx = 0
        for img, tag, points in train_loader:
            one_batch_time_start = time.time()

            img,tag = img.to(self.device), tag.to(self.device)
            out = self.net(img)
            loss = self.criterion(out, tag)

            total_loss += loss.item() 

            self.optimizer.zero_grad()#把梯度置零
            loss.backward() #计算梯度
            self.optimizer.step() #更新参数

            count += len(img)

            train_loss = total_loss/count 
            tp_count += self.eval_acc(out, tag, points)
            train_acc = tp_count/count

            one_batch_time = time.time() - one_batch_time_start
            batch_time+=one_batch_time
            eta = int((batch_time/(batch_idx+1))*(len(train_loader)-batch_idx-1))
            
            print_epoch = ''.join([' ']*(4-len(str(epoch+1))))+str(epoch+1)
            print_epoch_total = str(self.args.epochs)+''.join([' ']*(4-len(str(self.args.epochs))))
            print('\r',
                    '{}/{} [{}/{} ({:.0f}%)] - TIME : {}, loss: {:.4f}, acc: {:.4f}  LR: {:f},  {}/{}     '.format(
                    print_epoch, print_epoch_total, batch_idx * len(img), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), 
                    datetime.timedelta(seconds=eta),
                    train_loss,
                    train_acc,
                    self.optimizer.param_groups[0]["lr"],
                    tp_count,
                    count
                    ), 
                    end="",flush=True)
            batch_idx += 1

            #break

    def onVal(self, val_loader, epoch):
        self.net.eval()
        val_loss = 0
        correct = 0

        i = 0
        with torch.no_grad():
            for img, tag ,points in val_loader:
                img,tag = img.to(self.device), tag.to(self.device)
                out = self.net(img)

                val_loss += self.criterion(out, tag)
                correct += self.eval_acc(out, tag, points)

                if i == 0 : self.show_train_heatmap(out, tag)
                i += 1

            val_loss /= len(val_loader.dataset)
            self.val_acc =  correct / len(val_loader.dataset)
            #print(self.eval_acc(out, tag), len(img), correct, len(val_loader.dataset))
            

        print(' \n           [VAL] loss: {:.5f}, acc: {:.3f}%    {}/{}\n'.format(
            val_loss, 100. * self.val_acc, len(val_loader.dataset)-correct, len(val_loader.dataset)))
        
        self.scheduler.step()
        self.checkpoint(epoch)
        self.earlyStop(epoch)
        
    
    def show_train_heatmap(self, out, tag):
        if not self.args.flag_show : return

        output_path = self.args.save_path + self.args.name + '/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        _t = out[0][0] + out[0][1] + out[0][2] + out[0][3] 
        #_t =  out[0][0]+out[0][1]+out[0][2]+out[0][3]
        plt.imshow(_t.detach().cpu().numpy().squeeze())
        plt.savefig(output_path+'pic_pre.jpg', bbox_inches='tight', dpi=450)
        plt.show()

        _t = tag[0][0] + tag[0][1] + tag[0][2] + tag[0][3] 
        #_t =  tag[0][0]+tag[0][1]+tag[0][2]+tag[0][3]
        plt.imshow(_t.detach().cpu().numpy().squeeze())
        plt.savefig(output_path+'pic_src.jpg', bbox_inches='tight', dpi=450)
        plt.show()
    
    def eval_acc(self, out, tag, points):
        scale = int(self.args.size / 64)
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

            r0 = (o_idx1[1] * scale, o_idx1[0] * scale)
            r1 = (o_idx2[1] * scale, o_idx2[0] * scale)
            r2 = (o_idx3[1] * scale, o_idx3[0] * scale)
            r3 = (o_idx4[1] * scale, o_idx4[0] * scale)
            #print(r0,r1,r2,r3, target)
            #print('tag : ',target*256)
            #print('out : ', r0,r1,r2,r3)
            
            dis1 = pow(pow(r0[0] - target[0] *self.args.size,2) + pow(r0[1] - target[1] *self.args.size, 2), 0.5)
            dis2 = pow(pow(r1[0] - target[2] *self.args.size,2) + pow(r1[1] - target[3] *self.args.size, 2), 0.5)
            dis3 = pow(pow(r2[0] - target[4] *self.args.size,2) + pow(r2[1] - target[5] *self.args.size, 2), 0.5)
            dis4 = pow(pow(r3[0] - target[6] *self.args.size,2) + pow(r3[1] - target[7] *self.args.size, 2), 0.5)
            #print(dis1,dis2,dis3,dis4)

            if dis1<2.5*scale and dis2<2.5*scale and dis3<2.5*scale and dis4<2.5*scale :
                tp += 1
        return tp

    
    def earlyStop(self, epoch):
        ### earlystop
        if self.val_acc>self.early_stop_value:
            self.early_stop_value = self.val_acc
            self.early_stop_dist = 0

        self.early_stop_dist+=1
        if self.early_stop_dist>self.args.early_stop:
            self.best_epoch = epoch-self.args.early_stop+1
            print("[INFO] Early Stop with patient %d , best is Epoch - %d :%f" % (self.args.early_stop,self.best_epoch,self.early_stop_value))
            self.earlystop = True
        if  epoch+1==self.args.epochs:
            self.best_epoch = epoch-self.early_stop_dist+2
            print("[INFO] Finish trainging , best is Epoch - %d :%f" % (self.best_epoch,self.early_stop_value))
            self.earlystop = True
    
    def modelSave(self, save_name):
        torch.save(self.net.state_dict(), save_name)

    def checkpoint(self, epoch):
        output_path = self.args.save_path + self.args.name + '/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        save_name = '%s_e%d_%.5f.pth' % ('cmeter_hr_last',epoch+1,self.val_acc)
        if self.val_acc > self.early_stop_value:
            delete_files(output_path, 'cmeter_hr_best')
            save_name = '%s_e%d_%.5f.pth' % ('cmeter_hr_best',epoch+1,self.val_acc)
        else:
            delete_files(output_path, 'cmeter_hr_last')

        last_save_path = os.path.join(output_path, save_name)
        torch.save(self.net.state_dict(), last_save_path)

        #if self.val_acc > self.early_stop_value :
        #    val(self.args, self.device, self.net, self.args.val_txt, 1, None, 'train')
        #    onVal_test(self.args, self.device, self.net)#self.net

if __name__ == '__main__':
    torch.use_deterministic_algorithms(True)
    seed_everything(0)
    args = parse_args()

    runner = HRCmeter(args)

    runner.train()