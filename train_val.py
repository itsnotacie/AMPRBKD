#此代码位于darknet根目录下
from glob import glob
import cv2
import os
import shutil

import random
random.seed(0) 
import numpy as np
np.random.seed(0)

def save_img_file(path, train_type, txt_file):
    image_path = path + train_type
    image_path_list = os.listdir(image_path)
    #print(image_path_list)

    with open(txt_file, 'a') as tf:
        i=0
        for img_path in image_path_list:
            #if 'mask3' in img_path:
            #    continue

            if os.path.isfile(os.path.join(image_path,img_path)):
                continue
            img_path_ = image_path + img_path +'/'
            print(img_path_)
            for t_file in glob(img_path_ + '*.txt'):
                with open(t_file, "r") as f:
                    for line in f.readlines():
                        line = line.strip('\n')
                        listLine = line.split(' ')
                        listLine[0] = train_type + img_path +'/' + listLine[0]
                        #print(listLine)
                        string = ' '.join(listLine)
                        if '.jp' in string or '.png' in string: 
                            tf.writelines(string + '\n')
                            i += 1
                        #print(string)
                        #break
                        #txt_path.append(line)
    print(i)        
    tf.close()

def get_5000_data(path, train_txt,output_path):
    txt_file = path + train_txt

    txt_list = []
    with open(txt_file, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            txt_list.append(line)
    f.close()
    
    random.shuffle(txt_list)
    res_list = txt_list[0:5000]

    out_txt_file = output_path + train_txt

    with open(out_txt_file, 'a') as out_txt_file:
        for txt in res_list:
            listLine = txt.split(' ')
            img_path= path + listLine[0]
            out_img_path = output_path + listLine[0]

            #im = cv2.imread(img_path)
            #print(im.shape,img_path,out_img_path)
            #cv2.imwrite(out_img_path,im)
            shutil.copy(img_path, out_img_path)

            out_txt_file.write(txt+ '\n')
    out_txt_file.close()

if __name__ == '__main__':
    path = '../../cmeter_train/dial/'
    train_txt = 'hr_cmeter.txt'

    output_path = 'dial/'
    get_5000_data(path, train_txt,output_path)


    '''train_type = 'train/'
    test_type = 'test/'

    txt_file = 'dial/hr_cmeter.txt'
    if os.path.exists(txt_file) : os.remove(txt_file)
    save_img_file(path, train_type, txt_file)

    txt_file = 'dial/hr_cmeter_test.txt'
    if os.path.exists(txt_file) : os.remove(txt_file)
    save_img_file(path, test_type, txt_file)'''
