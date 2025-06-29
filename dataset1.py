# coding=utf-8
import os
import pandas as pd
import math
import torch.utils.data as Data
import cv2
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as T
from torchvision.utils import save_image
from transforms import *
from skimage.transform import resize
import random
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
import torch
from scipy.io import loadmat
from sklearn.model_selection import KFold
import time
from torch import Tensor
from typing import Tuple, List, Optional
import numbers
import shutil
from torchvision.utils import make_grid, save_image

import warnings
warnings.filterwarnings("ignore")

class Lung3D_eccv_patient_supcon(Data.Dataset):
    def __init__(self, train=False, val=False, na=False, inference=False, n_classes = 2, supcon=False, box_lung=False, seg_sth=None, iccv_test=False, add_mosmed=False):

        self.n_classes = n_classes
        self.img_size = 256
        self.img_depth = 64 #####
        self.train = train
        self.val = val
        self.inference = inference
        self.box_lung = box_lung
        self.seg_sth = seg_sth
        self.datalist = []
        self.supcon = supcon

        if self.supcon:
            if seg_sth:
                self.twocroptransform = TwoCropTransform2(augment2, ifhalfcrop=False) #####
            else:
                self.twocroptransform = TwoCropTransform(augment, ifhalfcrop=False) #####

        # self.root_dir = "./data/eccv/3d-data-norm/"
        self.root_dir = '/remote-home/share/21-yuanruntian-21210240410/cvpr24/challenge2/3d-norm/'

        if box_lung:
            train_path = 'getbox_train_new.csv'
            train_file = pd.read_csv(self.root_dir+train_path)
            self.train_dict = {}
            for index, row in train_file.iterrows():
                image_name = row['id']
                self.train_dict[image_name] = [row['top'],row['bottom'],row['left'],row['right'],row['front'],row['behind']]            

            val_path = 'getbox_val_new.csv'
            val_file = pd.read_csv(self.root_dir+val_path)
            self.val_dict = {}
            for index, row in val_file.iterrows():
                image_name = row['id']
                self.val_dict[image_name] = [row['top'],row['bottom'],row['left'],row['right'],row['front'],row['behind']]        

        if n_classes == 2:
            types = ['non-covid','covid']
            
            if train:
                # self.root_dir = '/remote-home/share/21-yuanruntian-21210240410/cvpr24/challenge12/3d-norm/'

                for i in range(len(types)):
                    for scan in sorted(os.listdir(self.root_dir+'train/'+types[i]+'/')):
                        img = os.path.join(self.root_dir+'train/'+types[i]+'/', scan)
                        name = types[i]+'/'+scan
                        if box_lung:
                            self.datalist.append((img, i, name, self.train_dict[name.split('.')[0]]))
                        else:
                            self.datalist.append((img, i, name))
                
                # add iccv test
                if iccv_test:
                    print('add iccv test')

                    for label in range(len(types)):
                        path = '/remote-home/share/21-yuanruntian-21210240410/da/pseudo/pseudo_labels_threshold_0.5_'+types[label]+'.csv'
                        file = pd.read_csv(path, header=None)
                        # imgs = file.loc[0]
                        # prob = file.loc[1]

                        # img_list=[]
                        # for i in range(len(prob)):
                        #     img_list.append((imgs[i],float(prob[i])))
                            
                        # img_list.sort(key=lambda x:x[1],reverse=True)
                        # select_num = 400 # 0.91
                        # print("add iccv test number: ", img_list[select_num][1])
                        for i in range(len(file)):
                            self.datalist.append(('/remote-home/share/21-yuanruntian-21210240410/cvpr24/challenge2/3d-norm/train/non-annotated/train/'+file[0][i]+'.npy', label, file[0][i]))

            if val:

                for i in range(len(types)):
                    for scan in sorted(os.listdir(self.root_dir+'val/'+types[i]+'/')):
                        img = os.path.join(self.root_dir+'val/'+types[i]+'/', scan)
                        name = types[i]+'/'+scan
                        if box_lung:
                            self.datalist.append((img, i, name, self.val_dict[name.split('.')[0]]))
                        else:
                            self.datalist.append((img, i, name))

            if inference:
                for scan in sorted(os.listdir(self.root_dir+'test/classification/')):
                    img = os.path.join(self.root_dir+'test/classification', scan)
                    name = scan
                    print(scan)
                    self.datalist.append((img, name, name))    

            if na:
                for scan in sorted(os.listdir(self.root_dir+'train/non-annotated/train/')):
                    img = os.path.join(self.root_dir+'train/non-annotated/train/', scan)
                    name = scan
                    if box_lung:
                        self.datalist.append((img, name, self.train_dict[name.split('.')[0]]))
                    else:
                        self.datalist.append((img, 0, name))  # na的label默认0，后面不用
        
        elif n_classes == 4:
            if train:
                csv_path = '/home/feng/hjl/eccv-submit/data/eccv/3d-data-norm/train_partition_covid_categories.csv'
                csv_file = pd.read_csv(csv_path)
                for iter, rows in csv_file.iterrows():
                    name = rows['Name;Category'].split(';')[0]
                    img = os.path.join(self.root_dir+'train/covid/', name+'.npy')
                    grade = rows['Name;Category'].split(';')[-1]
                    if box_lung:
                        self.datalist.append((img, int(grade)-1, name, self.train_dict['covid/'+name]))
                    elif seg_sth == 'lung':
                        self.datalist.append((img, int(grade)-1, name, '/home/feng/hjl/eccv-submit/segment/eccv_seg/severity/lungmask/train/'+name+'.npy'))
                    elif seg_sth == 'lesion':
                        self.datalist.append((img, int(grade)-1, name, '/home/feng/hjl/eccv-submit/segment/eccv_seg/severity/lesionmask27/train/'+name+'.npy'))

                    else:
                        self.datalist.append((img, int(grade)-1, name))
                
                if add_mosmed: # 684,125,45,2
                    m_path = '/home/feng/hjl/eccv-submit/data/MosMedData/MosMedData_3Dnpy_HU/3D_studies/'
                    for i in range(4):
                        file = 'CT-' + str(i+1) +'/'
                        num = len(os.listdir(m_path+file))
                        if i == 0:
                            num = 115
                        for j in range(num):
                            img = os.listdir(m_path+file)[j]
                            name = img.split('.')[0]
                            self.datalist.append((m_path+file+img, i, name))                   


            if val:
                csv_path = '/home/feng/hjl/eccv-submit/data/eccv/3d-data-norm/val_partition_covid_categories.csv'
                csv_file = pd.read_csv(csv_path)
                for iter, rows in csv_file.iterrows():
                    name = rows['Name;Category'].split(';')[0]
                    img = os.path.join(self.root_dir+'val/covid/', name+'.npy')
                    grade = rows['Name;Category'].split(';')[-1]
                    if box_lung:
                        self.datalist.append((img, int(grade)-1, name, self.val_dict['covid/'+name]))
                    elif seg_sth == 'lung':
                        self.datalist.append((img, int(grade)-1, name, '/home/feng/hjl/eccv-submit/segment/eccv_seg/severity/lungmask/val/'+name+'.npy'))
                    elif seg_sth == 'lesion':
                        self.datalist.append((img, int(grade)-1, name, '/home/feng/hjl/eccv-submit/segment/eccv_seg/severity/lesionmask/val/'+name+'.npy'))
                    else:
                        self.datalist.append((img, int(grade)-1, name))

            if inference:
                for scan in sorted(os.listdir(self.root_dir+'test/severity/')):
                    img = os.path.join(self.root_dir+'test/severity', scan)
                    name = scan
                    print(scan)
                    self.datalist.append((img, name, name))  

        print(len(self.datalist))
        


    def __getitem__(self, index):
        if self.box_lung:
            img, label, ID, lung = self.datalist[index]
            top, bottom, left, right, front, behind = lung
        elif self.seg_sth:
            img, label, ID, mask = self.datalist[index]
        else:
            img, label, ID = self.datalist[index]

        img_array = np.load(img) 
        if self.seg_sth:
            mask_array = np.load(mask)
            mask_array = mask_array.astype(np.uint8)

        # print(index,ID,img_array.shape,label)
        # save_path = '/remote-home/share/18-houjunlin-18110240004/iccvdata/iccv/3dlung/'
        # cv2.imwrite(save_path+'debug.png',img_array[20])



        if self.box_lung:
            # if right-left<128 and right<128:
            #     right = right+right-left
            # elif right-left<128 and left>128:
            #     left = left-(right-left)
            img_array = img_array[:,left:right,front:behind]
            img_array = rescale_gao(img_array)

        if 'study' in ID: #mosmed data
            img_array = normalize_hu(img_array)
            # img_array = rescale_z(img_array,self.img_depth)
            img_array = rescale_gao(img_array)


        # print(img_array.shape)
        if self.img_depth !=128:
            img_array = rescale_z(img_array,self.img_depth)
            if self.seg_sth:
                mask_array = rescale_z(mask_array,self.img_depth,is_mask_image=True)
                # img_array = img_array * lung_array
                # img_array = img_array.astype(np.uint8)

        # img_array = HU_transfer(img_array)


            # new = img_array[20]
            # cv2.imwrite('box_lung/train/'+ID+'.png',new)

        if self.train:
            if self.supcon:
                if self.seg_sth:
                    img_array, mask_array = self.twocroptransform(img_array, mask_array)
                else:
                    img_array = self.twocroptransform(img_array)
            else:
                img_array = augment(img_array, ifhalfcrop=False, ifrandom_resized_crop=True, ifflip=False, ifrotate=False, ifcontrast=True,ifswap = False,filling_value=0)
                img_array = torch.from_numpy(normalize(img_array))



            # new = img_array[0][20].numpy()
            # newmask = mask_array[0][20].numpy()
            # # cv2.imwrite('debug_20.png',new*255)
            # cv2.imwrite('lesioncat/img_noseed/2/'+ID+'.png',new*255)
            # cv2.imwrite('lesioncat/lesion_noseed/2/'+ID+'.png',newmask*255)

        if self.val: 
            #tta
            # img_array = augment(img_array, ifhalfcrop=False, ifrandom_resized_crop=True, ifflip=False, ifrotate=False, ifcontrast=True,ifswap = False,filling_value=0)
            img_array = torch.from_numpy(normalize(img_array)).float()
            if self.seg_sth:
                mask_array = torch.from_numpy(mask_array).float()
            # img_array = HU_transfer(img_array)

            # new = img_array[20].numpy()
            # cv2.imwrite('box_lung/val/'+ID+'.png',new*255)
            # cv2.imwrite('debug_20.png',new*255)

        if self.inference:
            #tta
            img_array = augment(img_array, ifhalfcrop=False, ifrandom_resized_crop=True, ifflip=False, ifrotate=False, ifcontrast=True,ifswap = False,filling_value=0)
            img_array = normalize(img_array)


        # new = img_array[20]
        # cv2.imwrite('./eccv_renorm.png',new*255)
        # if 'study' in ID:
        #     new = make_grid(torch.cat([torch.from_numpy(img_array1[20]), torch.from_numpy(img_array[20])/255], 0), nrow=2)
        #     # print(new.shape)
        #     # new = img_array1[0][20].numpy()
        #     new = new.permute(1,2,0)
        #     new = new.numpy()
            
        #     cv2.imwrite('mosmed_debug.png',new*255)
        
        
        # img_array = img_array.unsqueeze(1)
        # if self.seg_sth:
        #     mask_array = mask_array.unsqueeze(1)
        #     img_array = torch.cat([img_array, mask_array], dim=1) # 64,2,256,256

        '''binary cls'''
        # label = 1 if label == 1 else 0
        if self.seg_sth:
            return img_array, mask_array, label, ID
        return img_array, 0, label, ID       
    
    def __len__(self):
        return len(self.datalist)

def normalize_hu(image):
    MIN_BOUND = -1150
    MAX_BOUND = 350
    image[image < MIN_BOUND] = MIN_BOUND
    image[image > MAX_BOUND] = MAX_BOUND
    image = 1.0 * (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    return image *255


def HU_transfer(image): 
    # input image: (0~255)
    # output image: (0~1)

    # PNG denorm to HU
    eccv_min = -1150
    eccv_max = 350
    image = image / 255. * (eccv_max-eccv_min) + eccv_min

    # renorm to PNG
    MIN_BOUND = -1200
    MAX_BOUND = -100
    image[image < MIN_BOUND] = MIN_BOUND
    image[image > MAX_BOUND] = MAX_BOUND
    image = 1.0 * (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    return image*255




class Lung3D_ccii_patient_supcon(Data.Dataset):
    def __init__(self, train=False, val=False, inference=False, n_classes = 2):
        
        # self.n_classes = n_classes
        # self.img_size = 256
        # self.train = train
        # self.val = val
        # self.inference = inference
        # self.datalist = []
        # self.labelarraylist = []
        # self.twocroptransform = TwoCropTransform(augment)

        # '''cross validation'''
        # self.root_dir = "/remote-home/share/18-houjunlin-18110240004/cciidata/3DCC-CCII-norm/"
        # type_name = ['Normal','NCP','CP']
        # all_pat_dir = [[],[],[]] 

        # for i in range(len(type_name)):
        #     type = type_name[i]
        #     type_dir = self.root_dir + type + '/'
        #     for patient in sorted(os.listdir(type_dir)):
        #         all_pat_dir[i].append(patient)

        # NCP = []
        # CP = []
        # Normal = []
        # for i in range(len(type_name)):
        #     allpatdirs = all_pat_dir[i]
        #     if self.train:
        #         print(i, len(allpatdirs)) # all samples of a class

        #     KK = 0
        #     print("CV:",KK)
        #     kf = KFold(n_splits=5,shuffle=True,random_state=5)
        #     for k, (a,b) in enumerate(kf.split(range(len(allpatdirs)))):
        #         if k == KK:
        #             train_index, test_index = a, b
        #             if self.train:
        #                 print("before oversampling: ", len(train_index),len(test_index))
                        
        #     patient_index = train_index if train else test_index
            
        #     '''binary cls downsample'''
        #     if train and i!= 1:
        #         np.random.seed(0)
        #         patient_index = np.random.choice(patient_index,int(0.6*len(patient_index)),replace=False)
        #     for index in patient_index:
        #         patient = allpatdirs[index]
        #         for scan in sorted(os.listdir(self.root_dir+type_name[i]+'/'+patient+'/')):
        #             img = os.path.join(self.root_dir+type_name[i]+'/'+patient+'/', scan)
        #             name = type_name[i]+'/'+patient+'/'+scan
        #             self.datalist.append((img, i, name))

        #     print(len(self.datalist))




        
        self.n_classes = n_classes
        self.img_size = 256
        self.train = train
        self.val = val
        self.inference = inference
        self.datalist = []
        self.twocroptransform = TwoCropTransform(augment)

        self.root_dir = "/home/feng/hjl/eccv-submit/3d-data-norm/"
        types = ['non-covid','covid']
        
        if train:
            for i in range(len(types)):
                for scan in sorted(os.listdir(self.root_dir+'train/'+types[i]+'/')):
                    img = os.path.join(self.root_dir+'train/'+types[i]+'/', scan)
                    name = types[i]+'/'+scan
                    self.datalist.append((img, i, name))
        if val:
            for i in range(len(types)):
                for scan in sorted(os.listdir(self.root_dir+'val/'+types[i]+'/')):
                    img = os.path.join(self.root_dir+'val/'+types[i]+'/', scan)
                    name = types[i]+'/'+scan
                    self.datalist.append((img, i, name))
        if inference:
            for scan in sorted(os.listdir(self.root_dir+'iccvtest/')):
                img = os.path.join(self.root_dir+'iccvtest/', scan)
                name = scan
                print(scan)
                self.datalist.append((img, name, name))                 

        print(len(self.datalist))
        

    def __getitem__(self, index):

        img, label, ID = self.datalist[index]
        img_array = np.load(img) 
        # print(index,ID,img_array.shape,label)
        # save_path = '/remote-home/share/18-houjunlin-18110240004/iccvdata/iccv/3dlung/'
        # cv2.imwrite(save_path+'debug.png',img_array[20])

        

        if self.train:
            # img_array = rescale_z(img_array,80)
            img_array = self.twocroptransform(img_array)
            # img_array = augment(img_array, ifhalfcrop=False, ifrandom_resized_crop=True, ifflip=False, ifrotate=False, ifcontrast=True,ifswap = False,filling_value=0)

        if self.val: 
            #tta
            # img_array = augment(img_array, ifhalfcrop=False, ifrandom_resized_crop=True, ifflip=False, ifrotate=False, ifcontrast=True,ifswap = False,filling_value=0)
            img_array = normalize(img_array)
        if self.inference:
            #tta
            img_array = augment(img_array, ifhalfcrop=False, ifrandom_resized_crop=True, ifflip=False, ifrotate=False, ifcontrast=True,ifswap = False,filling_value=0)
            img_array = normalize(img_array)

        # save_path = '/remote-home/share/18-houjunlin-18110240004/iccvdata/iccv/3dlung/'
        # new = denormalize(img_array[0].numpy()[20])
        # cv2.imwrite(save_path+'debug.png',new)


        '''binary cls'''
        # label = 1 if label == 1 else 0

        return img_array, label, ID       
    
    def __len__(self):
        return len(self.datalist)




class Lung3D_ccii_patient_clf(Data.Dataset):
    def __init__(self, train=False, val=False, inference=False, n_classes = 2):
        self.n_classes = n_classes
        self.img_size = 256
        self.train = train
        self.inference = inference
        self.datalist = []

        # self.root_dir = "/remote-home/share/18-houjunlin-18110240004/iccvdata/3d-iccvdata-norm/"
        self.root_dir = "/remote-home/share/21-yuanruntian-21210240410/cvpr24/challenge2/3d-norm/"
        types = ['non-covid','covid']
        
        if train:
            for i in range(len(types)):
                for scan in sorted(os.listdir(self.root_dir+'train/'+types[i]+'/')):
                    img = os.path.join(self.root_dir+'train/'+types[i]+'/', scan)
                    name = types[i]+'/'+scan
                    self.datalist.append((img, i, name))
        if val:
            for i in range(len(types)):
                for scan in sorted(os.listdir(self.root_dir+'val/'+types[i]+'/')):
                    img = os.path.join(self.root_dir+'val/'+types[i]+'/', scan)
                    name = types[i]+'/'+scan
                    self.datalist.append((img, i, name))
        if inference:
            for scan in sorted(os.listdir(self.root_dir+'test/')):
                img = os.path.join(self.root_dir+'test/', scan)
                name = scan
                self.datalist.append((img, name, name))                 

        print(len(self.datalist))

    def __getitem__(self, index):

        img, label, ID = self.datalist[index]
        img_array = np.load(img) 
        # print(img_array.shape)
        # print(index,ID,img_array.shape,label)
        # save_path = '/remote-home/share/18-houjunlin-18110240004/iccvdata/iccv/3dlung/'
        # cv2.imwrite(save_path+'debug.png',img_array[20])

        

        if self.train:
            # img_array = rescale_z(img_array,80)
            img_array = augment(img_array, ifhalfcrop=False, ifrandom_resized_crop=True, ifflip=False, ifrotate=False, ifcontrast=True,ifswap = False,filling_value=0)

        img_array = torch.from_numpy(normalize(img_array))

        # save_path = '/remote-home/share/18-houjunlin-18110240004/iccvdata/iccv/3dlung/'
        # new = img_array[20]*255
        # new[new<70] = 70
        # new[new>170] = 170
        # new = (new-70)/170
        # cv2.imwrite(save_path+'debug.png',new*255)

        return img_array, label, ID       
    
    def __len__(self):
        return len(self.datalist)




def normalize(image):
    image = image / 255
    mean = 0.3529
    std = 0.2983
    # mean = 0.
    # std = 1.
    image = 1.0 * (image - mean) / std
    return image

def denormalize(image):
    mean = 0.3529
    std = 0.2983
    image = image * std + mean
    image = (image * 255).astype(np.uint8)
    return image

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform, ifhalfcrop):
        self.transform = transform
        self.ifhalfcrop = ifhalfcrop

    def __call__(self, x):
        return [ torch.from_numpy(normalize(self.transform(x, self.ifhalfcrop))).float(), torch.from_numpy(normalize(self.transform(x, self.ifhalfcrop))).float()]

class TwoCropTransform2:
    """Create two crops of the same image"""
    def __init__(self, transform, ifhalfcrop):
        self.transform = transform
        self.ifhalfcrop = ifhalfcrop

    def __call__(self, x, mask):
        x1, mask1 = self.transform(x, mask, ifhalfcrop=self.ifhalfcrop)
        x2, mask2 = self.transform(x, mask, ifhalfcrop=self.ifhalfcrop)

        x1 = normalize(x1)
        x2 = normalize(x2)

        x1 = torch.from_numpy(x1).float()
        mask1 = torch.from_numpy(mask1).float()
        x2 = torch.from_numpy(x2).float()
        mask2 = torch.from_numpy(mask2).float()


        return [ x1, x2], [mask1, mask2]


def rescale_z(images_zyx, target_depth, is_mask_image=False, verbose=False):
    # print("Resizing dim z")
    resize_x = 1.0
    resize_y = target_depth/images_zyx.shape[0]
    interpolation = cv2.INTER_NEAREST if is_mask_image else cv2.INTER_LINEAR
    res = cv2.resize(images_zyx, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)  # opencv assumes y, x, channels umpy array, so y = z pfff
    return res

def augment(sample, ifhalfcrop=True, ifrandom_resized_crop=True, ifflip=False, ifrotate=True, ifcontrast=True,ifswap = False,filling_value=0):
    if ifhalfcrop:
        # print(sample.shape[0]-64)
        start = np.random.randint(0,sample.shape[0]-64)
        sample = sample[start:start+64]
    if ifrandom_resized_crop:
        rrc = RandomResizedCrop(size=256)
        i,j,w,h = rrc(sample)
        sample = sample[:,i:i+w,j:j+h]
        sample = rescale_gao(sample)
    if ifrotate:
        # angle1 = np.random.rand()*45
        # # size = np.array(sample.shape[2:4]).astype('float')
        # rotmat = np.array([[np.cos(angle1/180*np.pi),-np.sin(angle1/180*np.pi)],[np.sin(angle1/180*np.pi),np.cos(angle1/180*np.pi)]])
        # sample = rotate(sample,angle1,axes=(1,2),reshape=False,cval=filling_value)
        angle = np.random.randint(-10, 10)
        sample = rotate(sample, angle, axes=(1,2), reshape=False, order=0, mode='reflect', cval=filling_value)
    if ifswap:
        if sample.shape[1]==sample.shape[2] and sample.shape[1]==sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample,np.concatenate([[0],axisorder+1]))
            # coord = np.transpose(coord,np.concatenate([[0],axisorder+1]))
    if ifflip:
        flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        sample = np.ascontiguousarray(sample[:,::flipid[0],::flipid[1]])
        # coord = np.ascontiguousarray(coord[:,::flipid[0],::flipid[1],::flipid[2]])
    if ifcontrast:
        contrast_low = 0.8
        contrast_up = 1.2
        brightness_low = -0.1
        brightness_up = 0.1
        c = np.random.uniform(contrast_low,contrast_up)
        b = np.random.uniform(brightness_low,brightness_up)
        sample = c * sample + b
        sample[sample>255] = 255
        sample[sample<0] = 0
    return sample


def augment2(sample, mask, ifstandardize=False, ifhalfcrop=False,ifrandom_resized_crop=True, ifflip=False, ifrotate=True, ifcontrast=True,ifswap = False,filling_value=0):
    filling_value = -1 if ifstandardize else 0

    if ifstandardize:
        mean = np.mean(sample)
        std = np.std(sample)
        sample = (sample - mean) / np.clip(std, a_min=1e-10, a_max=None) 
        print(sample.min(),sample.max())

    if ifhalfcrop:
        start = np.random.randint(0,sample.shape[0]-64)
        sample = sample[start:start+64]
        mask = mask[start:start+64]
    if ifrandom_resized_crop:
        rrc = RandomResizedCrop(size=256)
        i,j,w,h = rrc(sample)
        sample = sample[:,i:i+w,j:j+h]
        sample = rescale_gao(sample)
        mask = mask[:,i:i+w,j:j+h]
        mask = rescale_gao(mask, is_mask_image=True)

    if ifrotate:
        # angle = np.random.rand()*30
        # sample = rotate(sample,angle,axes=(1,2),reshape=False,cval=filling_value)
        angle = np.random.randint(-10, 10)
        sample = rotate(sample, angle, axes=(1,2), reshape=False, order=0, mode='reflect', cval=filling_value)
        mask = rotate(mask, angle, axes=(1,2), reshape=False, order=0, mode='reflect', cval=filling_value)

    if ifswap:
        if sample.shape[1]==sample.shape[2] and sample.shape[1]==sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample,np.concatenate([[0],axisorder+1]))
            # coord = np.transpose(coord,np.concatenate([[0],axisorder+1]))
    if ifflip:
        flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        sample = np.ascontiguousarray(sample[:,::flipid[0],::flipid[1]])
        # coord = np.ascontiguousarray(coord[:,::flipid[0],::flipid[1],::flipid[2]])
    if ifcontrast:
        contrast_low = 0.8
        contrast_up = 1.2
        brightness_low = -0.1
        brightness_up = 0.1
        c = np.random.uniform(contrast_low,contrast_up)
        b = np.random.uniform(brightness_low,brightness_up)
        sample = c * sample 
        sample[sample>255] = 255
        sample[sample<0] = 0
    return sample, mask


def rescale_gao(images_zyx, is_mask_image=False):
    res = images_zyx
    interpolation = cv2.INTER_NEAREST if is_mask_image else cv2.INTER_LINEAR

    if res.shape[0] > 512:
        res1 = res[:256]
        res2 = res[256:]
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res1 = cv2.resize(res1, dsize=(256, 256), interpolation=interpolation)
        res2 = cv2.resize(res2, dsize=(256, 256), interpolation=interpolation)
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res = np.vstack([res1, res2])
        # res = res.swapaxes(0, 2)
    else:
        res = cv2.resize(res.transpose(1, 2, 0), dsize=(256,256), interpolation=interpolation)
        res = res.transpose(2, 0, 1)
    # print("Shape after: ", res.shape)
    return res

class RandomResizedCrop(torch.nn.Module):
    """Crop the given image to random size and aspect ratio.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size (int or sequence): expected output size of each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a tuple or list of length 1, it will be interpreted as (size[0], size[0]).
        scale (tuple of float): range of size of the origin size cropped
        ratio (tuple of float): range of aspect ratio of the origin aspect ratio cropped.
        interpolation (int): Desired interpolation enum defined by `filters`_.
            Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
            and ``PIL.Image.BICUBIC`` are supported.
    """

    def __init__(self, size, scale=(0.7, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        super().__init__()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        elif isinstance(size, Sequence) and len(size) == 1:
            self.size = (size[0], size[0])
        else:
            if len(size) != 2:
                raise ValueError("Please provide only two dimensions (h, w) for size.")
            self.size = size

        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(
            img: Tensor, scale: Tuple[float, float], ratio: Tuple[float, float]
    ) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image or Tensor): Input image.
            scale (tuple): range of scale of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        _, width, height = img.shape
        area = height * width

        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(*scale).item()
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.
        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return i, j, h, w
        # return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


if __name__ == "__main__":
    dst = Lung3D_eccv_patient_supcon(train=True,n_classes=4,box_lung=True,supcon=True)
    # exit()
    for i in range(len(dst)):
        img_array, _, label, ID = dst.__getitem__(i) 
        
    exit()  
    from functions import get_mean_and_std
    mean, std = get_mean_and_std(dst)
    print(mean, std)


