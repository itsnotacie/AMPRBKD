import numpy as np
import json
from glob import glob
import matplotlib.pyplot as plt
import os
import cv2
import imgaug as ia

def flip_img(img, flip_method):
    return cv2.flip(img, 0)

def flip_pts(pts_lsit, img, flip_method):
    new_pts = []
    h,w,_ = img.shape
    for pt in pts_lsit:
        new_pts.append(ia.Keypoint(x = pt.x,y = h - pt.y))
    return new_pts

def random_hsv_transform(img, hue_vari = 0.5, sat_vari = 0.5, val_vari = 0.5):
    """
    :param img:
    :param hue_vari: 色调变化比例范围
    :param sat_vari: 饱和度变化比例范围
    :param val_vari: 明度变化比例范围
    :return:
    """
    hue_delta = np.random.randint(-hue_vari, hue_vari)
    sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
    val_mult = 1 + np.random.uniform(-val_vari, val_vari)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
    img_hsv[:, :, 1] *= sat_mult
    img_hsv[:, :, 2] *= val_mult
    img_hsv[img_hsv > 255] = 255
    return cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)

def affine_rotation_matrix(angle=(-20, 20)):
    """Create an affine transform matrix for image rotation.
    NOTE: In OpenCV, x is width and y is height.

    Parameters
    -----------
    angle : int/float or tuple of two int/float
        Degree to rotate, usually -180 ~ 180.
            - int/float, a fixed angle.
            - tuple of 2 floats/ints, randomly sample a value as the angle between these 2 values.

    Returns
    -------
    numpy.array
        An affine transform matrix.

    """
    if isinstance(angle, tuple):
        theta = np.pi / 180 * np.random.uniform(angle[0], angle[1])
    else:
        theta = np.pi / 180 * angle
    rotation_matrix = np.array([[np.cos(theta), np.sin(theta), 0], \
                                [-np.sin(theta), np.cos(theta), 0], \
                                [0, 0, 1]])
    return rotation_matrix

def affine_transform_cv2(x, flags=None, border_mode='constant'):
    rows, cols = x.shape[0], x.shape[1]
    o_x = (cols - 1) / 2.0
    o_y = (rows - 1) / 2.0
    if flags is None:
        flags = cv2.INTER_AREA
    if border_mode is 'constant':
        border_mode = cv2.BORDER_CONSTANT
    elif border_mode is 'replicate':
        border_mode = cv2.BORDER_REPLICATE
    else:
        raise Exception("unsupport border_mode, check cv.BORDER_ for more details.")
    x = cv2.warpAffine(x, transform_matrix[0:2,:], \
            (cols,rows), flags=flags, borderMode=border_mode)
    
    return x
           

def transform_matrix_offset_center(matrix, x, y):
    o_x = (x - 1) / 2.0
    o_y = (y - 1) / 2.0
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def affine_transform_keypoints(coords_list, transform_matrix):
    coords_result_list = []
    for coords in coords_list:
        coords = np.asarray(coords)
        coords = coords.transpose([1, 0])
        coords = np.insert(coords, 2, 1, axis=0)
        # print(coords)
        # print(transform_matrix)
        coords_result = np.matmul(transform_matrix, coords)
        coords_result = coords_result[0:2, :].transpose([1, 0])
        coords_result_list.append(coords_result)
    return coords_result_list

def CenterGaussianHeatMap(img_height, img_width, c_x, c_y, variance):
    gaussian_map = np.zeros((img_height, img_width))
    for x_p in range(img_width):
        for y_p in range(img_height):
            dist_sq = (x_p - c_x) * (x_p - c_x) + \
                      (y_p - c_y) * (y_p - c_y)
            exponent = dist_sq / 2.0 / variance / variance
            gaussian_map[y_p, x_p] = np.exp(-exponent)
    return gaussian_map

def CenterLabelHeatMap(img_width, img_height, c_x, c_y, sigma):
    X1 = np.linspace(1, img_width, img_width)
    Y1 = np.linspace(1, img_height, img_height)
    [X, Y] = np.meshgrid(X1, Y1)
    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)
    return heatmap

class GenerateHeatmap():
    def __init__(self, output_res, num_parts):
        self.output_res = output_res
        self.num_parts = num_parts
        sigma = self.output_res/18
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints):
        hms = np.zeros(shape = (self.num_parts, self.output_res, self.output_res), dtype = np.float32)
        sigma = self.sigma
        for p in keypoints:
            for idx, pt in enumerate(p):
                if pt[0] > 0: 
                    x, y = int(pt[0]), int(pt[1])
                    if x<0 or y<0 or x>=self.output_res or y>=self.output_res:
                        continue
                    ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
                    br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                    c,d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a,b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc,dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa,bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb,cc:dd] = np.maximum(hms[idx, aa:bb,cc:dd], self.g[a:b,c:d])
        return hms

d_kernel_size = [9,11,13,15]
def motion_blur(img, mode):
    kernel = None
    
    if mode == 0:
        kernel_size = np.random.randint(9,16)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[:, int((kernel_size -1)/2)] = np.ones(kernel_size)/kernel_size
    elif mode == 1:
        kernel_size = np.random.randint(9,16)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size -1)/2), :] = np.ones(kernel_size)/kernel_size
    elif mode == 2:
        kernel_size = d_kernel_size[np.random.randint(0,4)]
        kernel = np.eye(kernel_size)/kernel_size
    return cv2.filter2D(img, -1, kernel)
    
if __name__ == '__main__':
    f = open('datalist.txt')
    labels = f.readlines()
    labels = [label.strip().split(' ') for label in labels]
    hmgenerator = GenerateHeatmap(64,4)
    for label in labels:
        points = list(map(float,label[1:]))
        points = [64*x for x in points]
        print(points)
        img_name = label[0]
        
        img_path = os.path.join('data', img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img,(64,64))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kps = np.array(points).reshape(1,4,2)
        kps = kps.tolist()
        heatmap = hmgenerator(kps)
        # heatmap = []
        # for i in range(4):
        #     ptx, pty = points[2*i] * 64, points[2*i+1]*64
        #     heatmap.append(CenterLabelHeatMap(64,64,ptx,pty,1.5)) 
        gmap = heatmap[0] + heatmap[1] + heatmap[2] + heatmap[3] 
        cv2.imshow('o', img)
        plt.imshow(gmap)
        plt.show()
        cv2.waitKey(0)
