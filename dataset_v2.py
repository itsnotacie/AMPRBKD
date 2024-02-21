import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from utils import CenterGaussianHeatMap, CenterLabelHeatMap
from PIL import Image
import os
import cv2
import numpy as np
from utils import affine_rotation_matrix,affine_transform_cv2, affine_transform_keypoints, GenerateHeatmap, flip_img, flip_pts
#from motionblur import motion_blur
import imgaug.augmenters as iaa
import imgaug as ia
import matplotlib.pyplot as plt

import random
import numpy as np

size_t = 64 #64    #IMG_SIZE/4   #512-128  #256-64   #640-160
num_parts = 5
#IMG_SIZE = 256
IMG_SIZE = 256

num_class = 4#1,5,4

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ColorJitter(),
    transforms.ToTensor()
])

seq = iaa.Sequential([#iaa.Sometimes(0.5,iaa.Affine(scale={"x":(0.8,1.2),"y":(0.8,1.2)}, #缩放
                                #translate_percent={"x":(-0.2,0.2),"y":(-0.2,0.2)},#平移
                                # rotate=(-25, 25),#旋转
                                # shear=(-8,8))),#剪切
                        #iaa.Sometimes(0.5,iaa.Affine(scale={"x":(0.8,1.2),"y":(0.8,1.2)})),
                        #iaa.Sometimes(0.5,iaa.Crop(percent=(0,0.1))),#随机裁剪图片边长比例的0~0.1
                        #iaa.Sometimes(0.5,iaa.GaussianBlur(sigma=(0,0.5))), ##高斯模糊

                    #iaa.Fliplr(p=1, name=None,  deterministic="deprecated", random_state=None),
                    ])
#seq = iaa.Sequential([iaa.Affine(rotate=(-25, 25)),iaa.Grayscale(alpha = (0.0,1.0)),iaa.WithChannels(0, iaa.Add((0, 20))),iaa.WithChannels(1, iaa.Add((0,20))),iaa.WithChannels(2, iaa.Add((0,20)))])

import albumentations as A

'''def spot_blur(image, kernel_size= 30):
    kernel_h = np.zeros((kernel_size, kernel_size))
    kernel_v = np.copy(kernel_h)
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)

    kernel_v /= kernel_size
    kernel_h /= kernel_size

    vertical_mb = cv2.filter2D(image, -1, kernel_v)
    horizonal_mb = cv2.filter2D(image, -1, kernel_h)
    return horizonal_mb, vertical_mb'''


class HMDataset(Dataset):
    def __init__(self, txt, path, type):
        super(HMDataset,self).__init__()

        #seed_everything(0)

        f = open(txt)
        lines = f.readlines()

        self.prefix = path
        l = []
        for line in lines:
            label = line.strip().split()
            #img_path = os.path.join(label[0])
            img_path = os.path.join(path, label[0])
            if os.path.exists(img_path) : l.append(line)
        lines = l
        #print(lines)
        self.type = type

        self.lines = [line.strip().split() for line in lines]
        #self.prefix = path
        self.hmgenerator = GenerateHeatmap(size_t,num_parts)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        label = self.lines[idx]
        points = list(map(float,label[1:]))

        out_points = np.array(label[1:], dtype=float)
        
        img_path = os.path.join(self.prefix, label[0])

        #img_path = os.path.join(label[0])
        #print(img_path)
        img = cv2.imread(img_path)
        #print(img_path)
        #print(len(img))

        h,w,_ = img.shape
        
        kps = []
        assert w != 0 and h != 0
        for i in range(0,num_class):###5
            _x = points[2 * i] * w
            _y = points[2 * i + 1] * h
            #_x = points[i][0] * w
            #_y = points[i][1] * h
            kps.append(_x)
            kps.append(_y)
        kps = np.asarray(kps).reshape(num_class,1,2)###4
        kps = kps.astype(np.float32)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

        kps = kps.reshape(num_class,2)###4

        kps = kps.tolist()
        ia_kps = []
        for kp in kps:
            ia_kps.append((kp[0], kp[1]))
        
        if self.type == 'train':
            rect, ia_kps_ = self.aug_img(img, kps)
        else:
            rect = img
            ia_kps_ = kps

        '''i = 0
        while(i < 10):
            #img1, ia_kps_ = aug_img(rect, ia_kps)
            print('^^^^^^^^^^^^^^^',i,'^^^^^^^^^^^^^^^')
            if 1:
                rect = cv2.circle(rect,(int(ia_kps_[0][0]),int(ia_kps_[0][1])), 5, (255,0,255), -1)
                rect = cv2.circle(rect,(int(ia_kps_[1][0]),int(ia_kps_[1][1])), 5, (0,255,255), -1)
                rect = cv2.circle(rect,(int(ia_kps_[2][0]),int(ia_kps_[2][1])), 5, (255,255,0), -1)
                rect = cv2.circle(rect,(int(ia_kps_[3][0]),int(ia_kps_[3][1])), 5, (255,255,255), -1)
                rect = cv2.circle(rect,(int(ia_kps_[4][0]),int(ia_kps_[4][1])), 5, (255,0,0), -1)
                rect = cv2.circle(rect,(int(ia_kps_[5][0]),int(ia_kps_[5][1])), 5, (0,250,0), -1)
                rect = cv2.circle(rect,(int(ia_kps_[6][0]),int(ia_kps_[6][1])), 5, (0,0,255), -1)
                #if len(ia_kps_) >= 7: 
            name = img_path.split('/')[-1].split('.jpg')[0]
            out_path = 'images/output/'
            img_out_path = out_path + name + '_' + str(i) + '.jpg'
            if os.path.exists(img_out_path) :
                os.remove(img_out_path)
        
            cv2.imwrite(img_out_path, rect)
            i += 1
            break'''


        ##imgaug数据增强
        #kps = kps.tolist()
        #ia_kps = []
        #for kp in kps:
        #    ia_kps.append(ia.Keypoint(x=kp[0], y=kp[1])) 
        #rect, ia_kps = seq(image=img, keypoints=ia_kps)
        ##rect = img


        rect = rect.astype(np.uint8)
        # cv2.cvtColor(rect, cv2.COLOR_RGB2BGR)

        gmaps = []
        # img = Image.fromarray(img)
        for kp in ia_kps_:
            #ptx, pty = kp.x/w*size_t, kp.y/h*size_t
            ptx, pty = kp[0]/w*size_t, kp[1]/h*size_t
            # gmaps = self.hmgenerator(kps)
            gmaps.append(CenterGaussianHeatMap(size_t,size_t,ptx, pty,3))


        '''_t = gmaps[0] + gmaps[1] + gmaps[2] + gmaps[3] + gmaps[4]+ gmaps[5]+ gmaps[6]
        plt.axis("off")
        plt.imshow(_t)
        plt.savefig('weights/hmeter_0719/pic.jpg', bbox_inches='tight', dpi=450)
        #plt.show()
        plt.clf()'''

        '''if type == 'train':
            b_int = np.random.randint(0,10)
            if b_int < 3:
                rect = cv2.resize(rect,(64,64))'''
        
        rect = cv2.resize(rect,(IMG_SIZE,IMG_SIZE))

        rect = rect.transpose((2,0,1))
        data = torch.FloatTensor(rect)

        #print('on val',data.shape,data[0][0])
        return data, torch.FloatTensor(gmaps), out_points
    
    def pixel_data_aug(self, img):
        img = A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.1, 
                contrast_limit=0.1, p=0.5), 
                A.HueSaturationValue(hue_shift_limit=10, 
                    sat_shift_limit=10, val_shift_limit=10,  p=0.5)], 
                p=0.5)(image=img)['image']
        img = A.RGBShift(r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=0.5)(image=img)['image']
        img = A.OneOf([
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
                A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=1)
                ],
                p=1)(image=img)['image']
        
        #模糊
        img = A.OneOf([A.MotionBlur(p=1), 
                        A.MedianBlur(blur_limit=3, p=1), 
                        A.Blur(blur_limit=3, p=1)
                        ],
                        p=1)(image=img)['image']
        
        #降低图像色彩分辨率
        img = A.Posterize (num_bits=4, always_apply=False, p=1)(image=img)['image']

        #3d浮雕效果
        img = A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=False, p=1)(image=img)['image']

        #随机扰动亮度、对比度、饱和度、色度
        img = A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=1)(image=img)['image']

        #随机下采样再上采样   #生成数据
        img = A.Downscale(scale_min=0.75, scale_max=0.9, interpolation=None, always_apply=False, p=1)(image=img)['image']

        #随机加阴影RandomShadow
        img = A.RandomShadow (shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, always_apply=False, p=1)(image=img)['image']

        return img

    def pixel_data_aug_x(self, img):
        '''transformed_Fog = A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08,always_apply=False, p=1)(image=img)#随机雾化
        img = transformed_Fog["image"]

        #添加小白格子
        img = A.CoarseDropout(max_holes=8, max_height=10, 
                max_width=10,
                min_holes=5, fill_value=255, 
                mask_fill_value=0, p=0.5)(image=img)['image']'''
        
        img = A.OneOf([#
                #A.GridDistortion(num_steps=10,distort_limit=0.3,border_mode=4,always_apply=False, p=1),#网格失真
                #A.ElasticTransform(alpha = 1, sigma = 50,alpha_affine = 50,interpolation = 1,border_mode = 4,value = None,mask_value = None,always_apply = False,approximate = False,p = 1),#弹性变换
                A.OpticalDistortion(p=0.5),#畸变
                A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08,always_apply=False, p=0.5), #随机雾化
                A.CoarseDropout(max_holes=8, max_height=10, 
                    max_width=10,
                    min_holes=5, fill_value=255, 
                    mask_fill_value=0, p=0.5),
                ],
                p=1)(image=img)['image']

        return img


    def space_data_aug(self, img1, labels):
        #print(len(labels),labels)
        transform = A.Compose([
            ##A.IAAPerspective(scale=0.001,keep_size=True, p=1),#随机透视变换 0.2
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10,border_mode=cv2.BORDER_REPLICATE, p=1),#随机仿射变换 #0.0625， #0.2
            ],
            keypoint_params=A.KeypointParams(format='xy',remove_invisible=True, angle_in_degrees=True))
        
        for ii in labels:
            h,w,_ = img1.shape
            if ii[0] < 0 or ii[1] < 0 or ii[0] > w or ii[1] > h: 
                return img1, labels
                #print('1 : ',labels)
        transformed = transform(image=img1, keypoints=labels)  
        img = transformed['image']
        labels1 = transformed['keypoints']
        return img, labels1


    def space_data_aug_flip(self, img1, labels):
        transform = A.Compose([A.HorizontalFlip(p=1),
                            ],
                keypoint_params=A.KeypointParams(format='xy'),
                )
        for ii in labels:
            h,w,_ = img1.shape
            if ii[0] < 0 or ii[1] < 0 or ii[0] > w or ii[1] > h: 
                return img1, labels
                #print('1 : ',labels)
        transformed = transform(image=img1, keypoints=labels)  
        img = transformed['image']
        labels1 = transformed['keypoints']
        return img, labels1


    def aug_img(self, img, ia_kps):
        #img = cv2.imread(img_path)
        image = img

        if random.random()< 0.5: #0.5
            image = self.pixel_data_aug(image)
        
        if random.random()< 0.0 :
            image = self.pixel_data_aug_x(image)

        ia_kps_t = ia_kps
        if random.random()< 0.5 : #0.5
            #print('1 : ', ia_kps_t)
            img_, ia_kps_ = self.space_data_aug(image, ia_kps_t)
            #image = img_
            #ia_kps_t= ia_kps_
            if len(ia_kps_) == 4 :
                flag = True
                for kp_ in ia_kps_:
                    if kp_[0] < 0 or kp_[1] < 0 : flag = False
                if flag:
                    image = img_
                    ia_kps_t = ia_kps_
            #print('2 : ',ia_kps_t )
        #if not len(ia_kps_t) == 7: print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        
        if random.random()< 0.1: #0.1
            image_, ia_kps_t_ = self.space_data_aug_flip(image, ia_kps_t)
            if len(ia_kps_t_) == 4 :
                r0 = (ia_kps_t_[0][0],ia_kps_t_[0][1])
                r2 = (ia_kps_t_[2][0],ia_kps_t_[2][1])
                ia_kps_t_[0] = r2
                ia_kps_t_[2] = r0
                image = image_
                ia_kps_t = ia_kps_t_
        #if not len(ia_kps_t) == 7: print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        if image is None : 
            image = img
            ia_kps_t = ia_kps
        return image, ia_kps_t

if __name__ == '__main__':
    dataset = HMDataset('dial/hr_hmeter_test.txt', 'dial/')
    dataloader = DataLoader(dataset, batch_size=32,shuffle=True, num_workers=8, pin_memory=True)#batch_size=32,num_workers=8
    #while 1:
    for i in dataloader:
        print(i)
        break